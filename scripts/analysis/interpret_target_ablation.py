import os
import sys
import re
import json
import copy
import tempfile
import importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib

# Avoid Tkinter / thread UI issues on Windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from lifelines.utils import concordance_index as lifelines_concordance_index


# ----------------------------
# Path setup (CRITICAL)
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")

sys.path.insert(0, SRC_DIR)
os.chdir(PROJECT_DIR)


# ----------------------------
# Repo dirs
# ----------------------------
MODELS_DIR = Path("results/models")
OUT_ROOT = Path("results/analysis/ablation")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Data loaders
# ----------------------------
from data_loader import create_data_loaders
from data_loader_chembl import create_data_loaders_from_chembl_csv


# ----------------------------
# Helpers: model module loading
# ----------------------------
def _load_model_module(backbone: str):
    bb = backbone.lower()
    if bb == "gcn":
        candidates = [
            "multi_target_model_gcn_refactored",
            "multi_target_model_multiple_models_gcn",  # fallback if exists
        ]
    elif bb == "gine":
        candidates = [
            "multi_target_model_multiple_models_gine",
            "multi_target_model_multiple_models_gin",  # fallback
        ]
    else:
        raise ValueError(f"Unknown backbone '{backbone}' (expected gcn/gine).")

    last_err = None
    for name in candidates:
        try:
            m = importlib.import_module(name)
            if hasattr(m, "MultiTargetPredictor"):
                return m
        except Exception as e:
            last_err = e

    raise ImportError(f"Could not import model module for backbone={backbone}. Last error: {last_err}")


def infer_backbone_variant(ckpt_obj, ckpt_path: Path):
    stem = ckpt_path.stem.lower()
    backbone = None
    variant = None

    if isinstance(ckpt_obj, dict):
        backbone = ckpt_obj.get("backbone", None)
        variant = ckpt_obj.get("model_type_variant", None)

    if backbone is None:
        if "gine" in stem:
            backbone = "gine"
        elif "gcn" in stem:
            backbone = "gcn"

    if variant is None:
        if "quantum" in stem:
            variant = "quantum"
        elif "classical" in stem:
            variant = "classical"

    if backbone is None or variant is None:
        raise ValueError(f"Could not infer backbone/variant for checkpoint {ckpt_path.name}")

    return backbone, variant


def make_config_for_checkpoint(base_config_path: str, ckpt: dict) -> str:
    """
    Create a temp config that matches:
      - dataset_meta targets/n_targets + csv_path/use_chembl
      - model_config snapshot from checkpoint (critical for quantum shapes)
    """
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    cfg = copy.deepcopy(base_cfg)

    # ---- dataset_meta alignment ----
    ds = (ckpt.get("dataset_meta", {}) or {}) if isinstance(ckpt, dict) else {}
    target_names = ds.get("target_names", None)
    n_targets = ds.get("n_targets", None)

    if target_names is not None and n_targets is not None:
        cfg["targets"] = {t: t for t in target_names}
        cfg.setdefault("model", {})
        cfg["model"]["n_targets"] = int(n_targets)

    cfg.setdefault("data", {})
    use_chembl = ds.get("use_chembl", None)
    if use_chembl is not None:
        cfg["data"]["use_chembl"] = bool(use_chembl)

    if cfg["data"].get("use_chembl", False):
        csv_path = ds.get("csv_path", cfg["data"].get("csv_path", None))
        if csv_path is not None:
            cfg["data"]["csv_path"] = csv_path

    # ---- CRITICAL: merge checkpoint model_config into cfg["model"] ----
    if isinstance(ckpt, dict):
        mcfg = ckpt.get("model_config", None)
        if isinstance(mcfg, dict):
            cfg.setdefault("model", {})
            cfg["model"].update(mcfg)

    # write temp file
    fd, tmp_path = tempfile.mkstemp(prefix="ablate_modelcfg_", suffix=".json")
    Path(tmp_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return tmp_path



def make_test_loader_for_checkpoint(ckpt: dict, effective_config_path: str, batch_size: int, num_workers: int = 0):
    ds = (ckpt.get("dataset_meta", {}) or {}) if isinstance(ckpt, dict) else {}
    use_chembl = ds.get("use_chembl", None)
    csv_path = ds.get("csv_path", None)

    if use_chembl is None:
        with open(effective_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        use_chembl = bool(cfg.get("data", {}).get("use_chembl", False))
        csv_path = cfg.get("data", {}).get("csv_path", None)

    if use_chembl:
        if not csv_path:
            with open(effective_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            csv_path = cfg.get("data", {}).get("csv_path", None)
        if not csv_path:
            raise ValueError("use_chembl=True but csv_path missing (checkpoint + config).")

        _, _, test_loader, _ = create_data_loaders_from_chembl_csv(
            csv_path=str(csv_path),
            batch_size=batch_size,
            num_workers=num_workers,
            use_chembl=True,
        )
        return test_loader

    _, _, test_loader = create_data_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        config_path=effective_config_path,
    )
    return test_loader


# ----------------------------
# Metrics (measured-only, consistent with your test.py)
# ----------------------------
def pearson_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = torch.sqrt((yt**2).sum()) * torch.sqrt((yp**2).sum()) + 1e-8
    return float((yt * yp).sum() / denom)


def flatten_measured(pred, target, mask):
    m = mask.bool()
    return pred[m].detach().cpu(), target[m].detach().cpu()


def evaluate_model(model, loader, max_batches: int | None = None):
    model.eval()
    all_pred, all_tgt, all_mask = [], [], []

    for bi, batch in enumerate(tqdm(loader, leave=False, desc="eval")):
        if max_batches is not None and bi >= max_batches:
            break

        preds, _, _ = model(batch["molecules"])
        targets = batch["individual_affinities"].to(DEVICE)
        mask = batch.get("affinity_mask", None)
        if mask is None:
            mask = torch.ones_like(targets, device=DEVICE)
        else:
            mask = mask.to(DEVICE)

        all_pred.append(preds.detach())
        all_tgt.append(targets.detach())
        all_mask.append(mask.detach())

    preds = torch.cat(all_pred, dim=0)
    targets = torch.cat(all_tgt, dim=0)
    mask = torch.cat(all_mask, dim=0)

    flat_p, flat_t = flatten_measured(preds, targets, mask)

    mse = float(((flat_p - flat_t) ** 2).mean().item()) if flat_t.numel() else float("nan")
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")

    if flat_t.numel() >= 2:
        ci = float(lifelines_concordance_index(flat_t.numpy().astype(np.float64),
                                              flat_p.numpy().astype(np.float64)))
        pear = pearson_corr(flat_t, flat_p)
    else:
        ci = 0.5
        pear = 0.0

    # per-target
    T = targets.shape[1]
    per_target = []
    for i in range(T):
        p_i, t_i = flatten_measured(preds[:, i], targets[:, i], mask[:, i])
        if t_i.numel() < 2:
            per_target.append({"rmse": float("nan"), "pearson": 0.0, "ci": 0.5, "n": int(t_i.numel())})
        else:
            mse_i = float(((p_i - t_i) ** 2).mean().item())
            per_target.append({
                "rmse": float(np.sqrt(mse_i)),
                "pearson": pearson_corr(t_i, p_i),
                "ci": float(lifelines_concordance_index(t_i.numpy().astype(np.float64),
                                                       p_i.numpy().astype(np.float64))),
                "n": int(t_i.numel()),
            })

    return {
        "rmse_measured": rmse,
        "pearson_measured": pear,
        "ci_measured": ci,
        "measured_count_total": int(mask.sum().item()),
        "n_samples": int(targets.shape[0]),
        "n_targets": int(T),
        "per_target": per_target,
    }


# ----------------------------
# Memory-slot ablation (zero one target embedding vector)
# ----------------------------
def get_target_embedding_weight(model):
    """
    Returns the nn.Embedding weight tensor for the attention module, or None if not found.
    Works for BOTH:
      - ClassicalAttentionLayer: model.attention.target_embeddings
      - QuantumMultiTargetAttention wrapper: model.attention.quantum_attention.target_embeddings (common pattern)
    """
    # classical path
    if hasattr(model, "attention") and hasattr(model.attention, "target_embeddings"):
        return model.attention.target_embeddings.weight

    # quantum path (common in your codebase)
    if hasattr(model, "attention") and hasattr(model.attention, "quantum_attention"):
        qa = model.attention.quantum_attention
        if hasattr(qa, "target_embeddings"):
            return qa.target_embeddings.weight

    # sometimes quantum module is directly model.attention
    if hasattr(model, "attention") and hasattr(model.attention, "target_embeddings"):
        return model.attention.target_embeddings.weight

    return None


@torch.no_grad()
def run_memory_ablation(model, loader, target_names, out_dir: Path, max_batches: int | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # baseline
    baseline = evaluate_model(model, loader, max_batches=max_batches)

    W = get_target_embedding_weight(model)
    if W is None:
        raise RuntimeError("Could not locate target_embeddings.weight in the model attention module.")

    T = W.shape[0]
    if T != len(target_names):
        # still proceed, but warn in the saved json
        pass

    ablations = []
    W_backup = W.detach().clone()

    for j in range(T):
        # zero out memory slot j
        W.data[j].zero_()

        metrics_j = evaluate_model(model, loader, max_batches=max_batches)

        # restore
        W.data.copy_(W_backup)

        ablations.append({
            "memory_slot": int(j),
            "memory_name": target_names[j] if j < len(target_names) else f"slot_{j}",
            "metrics": metrics_j,
            "delta": {
                "rmse_measured": float(metrics_j["rmse_measured"] - baseline["rmse_measured"]),
                "ci_measured": float(metrics_j["ci_measured"] - baseline["ci_measured"]),
                "pearson_measured": float(metrics_j["pearson_measured"] - baseline["pearson_measured"]),
            }
        })

    # save json
    payload = {
        "baseline": baseline,
        "ablations": ablations,
    }
    (out_dir / "ablation_memory_zero.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # plot: delta RMSE and delta CI by memory slot
    labels = [a["memory_name"] for a in ablations]
    d_rmse = [a["delta"]["rmse_measured"] for a in ablations]
    d_ci = [a["delta"]["ci_measured"] for a in ablations]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, d_rmse)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Δ RMSE (measured-only)")
    plt.title("Memory-slot ablation (zero embedding): impact on RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_rmse_by_memory.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(labels, d_ci)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Δ CI (measured-only)")
    plt.title("Memory-slot ablation (zero embedding): impact on CI")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_ci_by_memory.png", dpi=200)
    plt.close()

    return payload


# ----------------------------
# Load model from checkpoint
# ----------------------------
def build_model_from_checkpoint(ckpt_path: Path, effective_config_path: str):
    ckpt_obj = torch.load(ckpt_path, map_location=DEVICE)

    backbone, variant = infer_backbone_variant(ckpt_obj, ckpt_path)

    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state = ckpt_obj["model_state_dict"]
    else:
        state = ckpt_obj  # old style

    module = _load_model_module(backbone)
    ModelClass = module.MultiTargetPredictor

    model = ModelClass(config_path=effective_config_path, model_type=variant).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, backbone, variant, ckpt_obj


def load_target_names_from_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return list(cfg.get("targets", {}).keys())


# ----------------------------
# MAIN: run ablation for every checkpoint
# ----------------------------
def main(
    config_path: str = "config_ui_run.json",
    max_batches: int | None = None,
    num_workers: int = 0,
):
    model_files = sorted(MODELS_DIR.glob("*.pt"))
    if not model_files:
        print(f"No checkpoints found in {MODELS_DIR.resolve()}")
        return

    print(f"Found {len(model_files)} checkpoints in {MODELS_DIR.resolve()}")

    summary = []

    for ckpt_path in model_files:
        print(f"\n=== {ckpt_path.name} ===")

        # load checkpoint first (to know dataset/backbone/variant)
        ckpt_obj = torch.load(ckpt_path, map_location=DEVICE)
        backbone, variant = infer_backbone_variant(ckpt_obj, ckpt_path)

        # batch size: use checkpoint training_config.bs if available, else config file
        bs = None
        if isinstance(ckpt_obj, dict):
            bs = (ckpt_obj.get("training_config", {}) or {}).get("batch_size", None)
        if bs is None:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            bs = int(cfg.get("training", {}).get("batch_size", 16))

        # create effective config matched to checkpoint targets + dataset
        effective_config_path = make_config_for_checkpoint(config_path, ckpt_obj if isinstance(ckpt_obj, dict) else {})
        target_names = load_target_names_from_config(effective_config_path)

        # create correct test loader
        test_loader = make_test_loader_for_checkpoint(
            ckpt=ckpt_obj if isinstance(ckpt_obj, dict) else {},
            effective_config_path=effective_config_path,
            batch_size=int(bs),
            num_workers=num_workers,
        )

        # build model
        model, backbone2, variant2, ckpt_obj2 = build_model_from_checkpoint(ckpt_path, effective_config_path)
        assert backbone2 == backbone and variant2 == variant

        # output dir
        out_dir = OUT_ROOT / backbone / variant / ckpt_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # run ablation
        payload = run_memory_ablation(
            model=model,
            loader=test_loader,
            target_names=target_names,
            out_dir=out_dir,
            max_batches=max_batches,
        )

        # record summary
        baseline = payload["baseline"]
        summary.append({
            "checkpoint": ckpt_path.name,
            "backbone": backbone,
            "variant": variant,
            "baseline_rmse": baseline["rmse_measured"],
            "baseline_ci": baseline["ci_measured"],
            "baseline_pearson": baseline["pearson_measured"],
            "out_dir": str(out_dir),
        })

        # cleanup temp config
        try:
            if effective_config_path != config_path:
                Path(effective_config_path).unlink(missing_ok=True)
        except Exception:
            pass

        print(f"✓ Saved ablation results to {out_dir}")

    (OUT_ROOT / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDONE. Summary saved to {OUT_ROOT / 'ablation_summary.json'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_ui_run.json", help="Base UI config")
    ap.add_argument("--max_batches", type=int, default=None, help="Limit eval batches for speed (debug)")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    main(
        config_path=args.config,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )
