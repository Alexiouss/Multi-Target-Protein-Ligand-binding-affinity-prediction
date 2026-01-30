import os
import sys
import json
import argparse
import importlib
from pathlib import Path
import tempfile
import copy


import numpy as np
import torch
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_n_targets_from_state_dict(state: dict) -> int:
    # Most reliable for your architecture:
    # attention.target_embeddings.weight : [n_targets, feature_dim]
    k = "attention.target_embeddings.weight"
    if k in state:
        return int(state[k].shape[0])

    # fallback: count predictor heads if present (target_predictors.0..., 1..., etc)
    heads = set()
    for key in state.keys():
        if key.startswith("target_predictors."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                heads.add(int(parts[1]))
    if heads:
        return max(heads) + 1

    raise ValueError("Could not infer n_targets from checkpoint state_dict.")


def make_temp_config_matching_checkpoint(base_config_path: str, ckpt_obj: dict, project_root: str) -> str:
    """
    Creates a temp config that matches checkpoint targets:
      - n_targets
      - targets dict (and ordering)
      - data.use_chembl + csv_path if stored in checkpoint
    Returns temp config path.
    """
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)
    cfg = copy.deepcopy(base_cfg)

    # get state dict
    state = ckpt_obj["model_state_dict"] if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj else ckpt_obj
    # --- infer PCA dim from attention head input ---
    head_k = "attention.quantum_attention.attention_head.0.weight"
    if head_k in state:
        in_dim = int(state[head_k].shape[1])     # 6 in ckpt
        pca_dim = in_dim // 2                    # 3
        cfg.setdefault("quantum", {})
        cfg["quantum"]["pca_dim"] = int(pca_dim)
        # if your code stores it under model as well, keep both:
        cfg.setdefault("model", {})
        cfg["model"]["pca_dim"] = int(pca_dim)

    # --- infer n_qubits from quantum param vector length ---
    qp_k = "attention.quantum_attention.quantum_params.0"
    if qp_k in state:
        n_params = int(state[qp_k].numel())      # 36 in ckpt

        # Heuristic: many circuits use 9 params per qubit per layer (e.g., 3 rotations × 3 blocks)
        # 36 -> 4 qubits, 72 -> 8 qubits
        if n_params % 9 == 0:
            n_qubits = n_params // 9
            cfg.setdefault("quantum", {})
            cfg["quantum"]["n_qubits"] = int(n_qubits)
            cfg.setdefault("model", {})
            cfg["model"]["n_qubits"] = int(n_qubits)
        else:
            # fallback: at least store the raw count so you can debug in logs
            cfg.setdefault("quantum", {})
            cfg["quantum"]["quantum_params_len"] = int(n_params)


    # try to get metadata (new checkpoints)
    ds = {}
    if isinstance(ckpt_obj, dict):
        ds = ckpt_obj.get("dataset_meta", {}) or {}

    # target names & n_targets from checkpoint if available
    target_names = ds.get("target_names", None)
    n_targets = ds.get("n_targets", None)

    if n_targets is None:
        n_targets = _infer_n_targets_from_state_dict(state)

    if target_names is None:
        # if we don't know names, make deterministic placeholders
        target_names = [f"T{i}" for i in range(int(n_targets))]

    # override targets to match checkpoint order exactly
    cfg.setdefault("targets", {})
    cfg["targets"] = {t: t for t in target_names}

    cfg.setdefault("model", {})
    cfg["model"]["n_targets"] = int(n_targets)

    # also align dataset mode if checkpoint says so
    cfg.setdefault("data", {})
    if "use_chembl" in ds:
        cfg["data"]["use_chembl"] = bool(ds["use_chembl"])

    if cfg["data"].get("use_chembl", False):
        # use checkpoint csv_path if exists, else fallback to base config, else default repo path
        csv_path = ds.get("csv_path") or cfg["data"].get("csv_path")
        if not csv_path:
            csv_path = os.path.join(project_root, "data", "chembl", "chembl_affinity_dataset.csv")
        cfg["data"]["csv_path"] = str(csv_path)

    # write temp file
    fd, tmp_path = tempfile.mkstemp(prefix="extract_cfg_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return tmp_path


def _add_src_to_path(project_root: str):
    src_dir = os.path.join(project_root, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)

def _load_model_module(backbone: str):
    bb = backbone.lower()
    if bb == "gcn":
        module_name = "multi_target_model_gcn_refactored"
    elif bb == "gine":
        module_name = "multi_target_model_multiple_models_gine"
    else:
        raise ValueError(f"Unknown backbone '{backbone}' (expected gcn/gine)")
    return importlib.import_module(module_name)

def _build_loaders_from_config(project_root: str, config_path: str, split: str, override_bs: int = None):
    """
    Loaders API:
      - synthetic: create_data_loaders(batch_size, num_workers, config_path=...)
      - chembl: create_data_loaders_from_chembl_csv(csv_path, batch_size, num_workers, use_chembl=True)
    """
    _add_src_to_path(project_root)

    from data_loader import create_data_loaders
    from data_loader_chembl import create_data_loaders_from_chembl_csv

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bs = int(override_bs if override_bs is not None else cfg["training"]["batch_size"])
    nw = int(cfg["training"].get("num_workers", 0))
    data_cfg = cfg.get("data", {})
    use_chembl = bool(data_cfg.get("use_chembl", False))

    if use_chembl:
        csv_path = data_cfg.get("csv_path", None)
        if not csv_path:
            # fallback: repo default (same as your train.py)
            csv_path = os.path.join(project_root, "data", "chembl", "chembl_affinity_dataset.csv")

        # returns (train, val, test, label_stats)
        tr, va, te, _ = create_data_loaders_from_chembl_csv(
            csv_path=str(csv_path),
            batch_size=bs,
            num_workers=nw,
            use_chembl=True,
        )
    else:
        # returns (train, val, test)
        tr, va, te = create_data_loaders(
            batch_size=bs,
            num_workers=nw,
            config_path=config_path,
        )

    if split == "train":
        return tr
    if split == "val":
        return va
    if split == "test":
        return te
    raise ValueError("split must be one of: train/val/test")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backbone", required=True, choices=["gcn", "gine"])
    ap.add_argument("--model_type", required=True, choices=["quantum", "classical"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--override_bs", type=int, default=None, help="Optional: force batch size for loader")
    args = ap.parse_args()

    project_root = os.path.abspath(args.project_root)
    config_path = args.config if os.path.isabs(args.config) else os.path.join(project_root, args.config)
    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(project_root, args.checkpoint)
    out_npz = args.out_npz if os.path.isabs(args.out_npz) else os.path.join(project_root, args.out_npz)

    _add_src_to_path(project_root)

    ckpt_obj = torch.load(ckpt_path, map_location=DEVICE)

    # ✅ create temp config matching checkpoint targets BEFORE model creation
    effective_config_path = make_temp_config_matching_checkpoint(
        base_config_path=config_path,
        ckpt_obj=ckpt_obj if isinstance(ckpt_obj, dict) else {"model_state_dict": ckpt_obj},
        project_root=project_root,
    )

    # build loader using effective config (so targets dimensions match)
    loader = _build_loaders_from_config(
        project_root=project_root,
        config_path=effective_config_path,
        split=args.split,
        override_bs=args.override_bs,
    )

    # build model using effective config
    module = _load_model_module(args.backbone)
    ModelClass = getattr(module, "MultiTargetPredictor")
    model = ModelClass(config_path=effective_config_path, model_type=args.model_type).to(DEVICE)

    state = ckpt_obj["model_state_dict"] if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj else ckpt_obj
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---- collect ----
    all_emb = []
    all_pred = []
    all_tgt = []
    all_mask = []
    all_smiles = []

    for bi, batch in enumerate(tqdm(loader, desc="Extract", leave=False)):
        if args.max_batches is not None and bi >= args.max_batches:
            break

        smiles = batch["molecules"]
        preds, attn, mol_feats = model(smiles)  # preds [B,T], mol_feats [B,256]
        targets = batch["individual_affinities"].to(DEVICE)
        mask = batch.get("affinity_mask", None)
        if mask is None:
            mask = torch.ones_like(targets, device=DEVICE)
        else:
            mask = mask.to(DEVICE)

        all_smiles.extend(list(smiles))
        all_emb.append(mol_feats.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())
        all_tgt.append(targets.detach().cpu().numpy())
        all_mask.append(mask.detach().cpu().numpy())

    emb = np.concatenate(all_emb, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    tgt = np.concatenate(all_tgt, axis=0)
    msk = np.concatenate(all_mask, axis=0)

    Path(os.path.dirname(out_npz)).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        smiles=np.array(all_smiles, dtype=object),
        embeddings=emb,
        predictions=pred,
        targets=tgt,
        mask=msk,
        meta=np.array(json.dumps({
            "checkpoint": ckpt_path,
            "config": config_path,
            "split": args.split,
            "backbone": args.backbone,
            "model_type": args.model_type,
            "n_samples": int(emb.shape[0]),
            "embedding_dim": int(emb.shape[1]),
        }), dtype=object),
    )

    print(f"✅ Saved: {out_npz} | embeddings={emb.shape}, preds={pred.shape}")

if __name__ == "__main__":
    main()
