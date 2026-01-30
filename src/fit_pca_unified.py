# fit_pca_unified.py
import os
import json
import hashlib
import importlib
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from data_loader import create_data_loaders
from data_loader_chembl import create_data_loaders_from_chembl_csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_module(backbone: str):
    bb = backbone.lower()
    if bb == "gcn":
        module_name = "multi_target_model_gcn_refactored"
    elif bb == "gine":
        module_name = "multi_target_model_multiple_models_gine"
    else:
        raise ValueError(f"Unknown backbone '{backbone}', expected 'gcn' or 'gine'.")

    module = importlib.import_module(module_name)
    required = ["MultiTargetPredictor"]
    for r in required:
        if not hasattr(module, r):
            raise AttributeError(f"Module '{module_name}' missing '{r}'.")
    return module


@torch.no_grad()
def collect_combined_vectors(model, train_loader, max_batches: int, n_targets: int):
    """
    Collect X of shape [N, 2*target_dim] where each row is [query || key]
    """
    model.eval()

    qa = model.attention.quantum_attention
    target_embeddings = qa.target_embeddings

    chunks = []

    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break

        smiles_list = batch["molecules"]

        mol_feats, _ = model.graph_encoder(smiles_list)
        mol_feats = mol_feats.to(DEVICE)

        queries = qa.query_projection(mol_feats)
        queries = qa.query_norm(queries)  # [B, D]

        # keys: [T, D]
        keys = []
        for t in range(n_targets):
            tid = torch.tensor([t], device=DEVICE)
            emb = target_embeddings(tid).squeeze(0)  # [D]
            key = qa.key_projection(emb.unsqueeze(0)).squeeze(0)
            key = qa.key_norm(key)
            keys.append(key)
        keys = torch.stack(keys, dim=0)

        B, D = queries.shape
        q_exp = queries.unsqueeze(1).expand(-1, n_targets, -1)  # [B,T,D]
        k_exp = keys.unsqueeze(0).expand(B, -1, -1)             # [B,T,D]

        combined = torch.cat([q_exp, k_exp], dim=-1)            # [B,T,2D]
        combined = combined.reshape(-1, combined.shape[-1])     # [B*T,2D]
        chunks.append(combined.cpu().numpy())

    if not chunks:
        raise RuntimeError("No PCA data collected (train_loader empty or max_batches=0).")

    X = np.concatenate(chunks, axis=0)
    return X


def ensure_pca_for_config(
    config_path: str,
    backbone: str,
    max_batches: int = 50,
    random_state: int = 42,
):
    # ---- Load config ----
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    encoding_type = model_cfg.get("encoding_type", "").lower()
    n_qubits = int(model_cfg.get("n_qubits", 0))
    target_dim = int(model_cfg.get("feature_dim", 0))
    n_targets = int(model_cfg.get("n_targets", 0))

    if target_dim <= 0 or n_targets <= 0:
        raise ValueError("Config must include model.feature_dim and model.n_targets > 0.")

    input_dim = 2 * target_dim

    # ---- Decide PCA output dimensionality + filename ----
    if encoding_type == "amplitude":
        out_dim = 16
        pca_path = model_cfg.get("pca_path", "pca_amplitude.npz")
    elif encoding_type == "angle":
        out_dim = 2 * n_qubits
        pca_path = f"pca_angle_{n_qubits}.npz"
    else:
        print(f"⚠️ Skipping PCA: encoding_type='{encoding_type}' not supported.")
        return None

    # ---- Build loaders like training does ----
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))

    use_chembl = bool(data_cfg.get("use_chembl", False))
    if use_chembl:
        csv_path = data_cfg.get("csv_path", None)
        if not csv_path:
            raise ValueError("data.use_chembl=True but data.csv_path is missing.")
        train_loader, _, _, _ = create_data_loaders_from_chembl_csv(
            csv_path=str(csv_path),
            batch_size=batch_size,
            num_workers=num_workers,
            use_chembl=True,
        )
    else:
        train_loader, _, _ = create_data_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            config_path=config_path,
        )

    # ---- Signature to prevent stale PCA ----
    signature = {
        "version": 1,
        "encoding_type": encoding_type,
        "n_qubits": n_qubits,
        "target_dim": target_dim,
        "input_dim": input_dim,
        "out_dim": out_dim,
        "n_targets": n_targets,
        "batch_size": batch_size,
        "synthetic_samples": int(data_cfg.get("synthetic_samples", -1)),
        "use_chembl": use_chembl,
        "max_batches": max_batches,
        "random_state": random_state,
        "backbone": backbone.lower(),  # gcn vs gine
    }

    sig_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()

    # ---- If PCA exists and signature matches, do nothing ----
    if os.path.exists(pca_path):
        try:
            old = np.load(pca_path, allow_pickle=True)
            meta = json.loads(old["meta"].item())
            if meta.get("sig_hash") == sig_hash:
                print(f"✓ PCA already valid → {pca_path}")
                return pca_path
        except Exception:
            pass  # treat as stale/corrupt

    print(f"🔁 Computing PCA → {pca_path}")
    print(f"   encoding={encoding_type}, backbone={backbone}, input_dim={input_dim}, out_dim={out_dim}")

    # ---- Build temp model with chosen backbone ----
    module = _load_model_module(backbone)
    ModelClass = module.MultiTargetPredictor
    model = ModelClass(config_path=config_path, model_type="quantum").to(DEVICE)

    # ---- Collect + fit PCA ----
    X = collect_combined_vectors(model, train_loader, max_batches=max_batches, n_targets=n_targets)

    if X.shape[1] != input_dim:
        raise RuntimeError(f"PCA input_dim mismatch: got {X.shape[1]}, expected {input_dim}")

    pca = PCA(n_components=out_dim, random_state=random_state)
    pca.fit(X)

    meta = dict(signature)
    meta["sig_hash"] = sig_hash
    meta["explained_var_sum"] = float(pca.explained_variance_ratio_.sum())

    np.savez(
        pca_path,
        components=pca.components_,
        mean=pca.mean_,
        meta=np.array(json.dumps(meta), dtype=object),
    )

    print(f"💾 Saved PCA → {pca_path} | explained_var_sum={meta['explained_var_sum']:.4f}")
    return pca_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_ui_run.json", help="Merged UI config path.")
    ap.add_argument("--backbone", default="gcn", choices=["gcn", "gine"], help="Graph encoder backbone.")
    ap.add_argument("--max_batches", type=int, default=50)
    args = ap.parse_args()

    ensure_pca_for_config(
        config_path=args.config,
        backbone=args.backbone,
        max_batches=args.max_batches,
    )
