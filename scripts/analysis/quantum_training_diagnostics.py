import os
import sys
import json
import glob
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt
import pennylane as qml

# -----------------------------
# CPU ONLY (avoid device mismatch with lightning.qubit outputs)
# -----------------------------
DEVICE = torch.device("cpu")

# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../Thesis_final
SRC_DIR = os.path.join(PROJECT_DIR, "src")
RESULTS_MODELS_DIR = os.path.join(PROJECT_DIR, "results", "models")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

print(f"✓ PROJECT_DIR: {PROJECT_DIR}")
print(f"✓ SRC_DIR: {SRC_DIR}")
print(f"✓ RESULTS_MODELS_DIR: {RESULTS_MODELS_DIR}")
print(f"✓ DEVICE: {DEVICE}")

# -----------------------------
# Import model
# -----------------------------
try:
    from multi_target_model_gcn_refactored import MultiTargetPredictor
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Could not import 'multi_target_model_gcn_refactored'.\n"
        "Make sure it exists in src/ and that scripts/ adds src/ to sys.path.\n"
        f"Expected: {os.path.join(SRC_DIR, 'multi_target_model_gcn_refactored.py')}"
    ) from e

# -----------------------------
# Utils
# -----------------------------
def find_latest_best_checkpoint(models_dir: str) -> str:
    cands = glob.glob(os.path.join(models_dir, "*_best.pt"))
    if not cands:
        raise FileNotFoundError(f"No '*_best.pt' checkpoints found in: {models_dir}")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing 'model_state_dict'.")
    return ckpt

def make_temp_config_from_ckpt(ckpt: dict, fallback_config_path: str) -> str:
    """
    Create a temp config that matches the checkpoint's quantum hyperparams
    (avoids size mismatch when instantiating the model).
    """
    if ckpt.get("config_path") and os.path.exists(ckpt["config_path"]):
        base_config_path = ckpt["config_path"]
    else:
        base_config_path = fallback_config_path

    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("model", {})
    ckpt_model_cfg = ckpt.get("model_config", None)

    if isinstance(ckpt_model_cfg, dict):
        for k in [
            "n_qubits", "n_layers", "feature_dim", "encoding_type",
            "use_data_reuploading", "pca_path", "angle_pca_path", "n_targets"
        ]:
            if k in ckpt_model_cfg:
                model_cfg[k] = ckpt_model_cfg[k]
        cfg["model"] = model_cfg

    fd, tmp_path = tempfile.mkstemp(prefix="ckpt_config_", suffix=".json", dir=os.path.dirname(__file__))
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"✓ Temp config created: {tmp_path}")
    return tmp_path

# -----------------------------
# Build a lightning.qubit circuit
# (expects z already in POST-PCA latent space: length = 2*n_qubits)
# -----------------------------
def build_angle_qnode_lightning(n_qubits: int, n_layers: int, use_reupload: bool):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    def entangle(layer):
        if layer % 2 == 0:
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        else:
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        if layer > 0:
            for q in range(0, n_qubits - 1, 2):
                qml.CZ(wires=[q, q + 1])

    def angle_encode_from_latent(z_2n):
        for q in range(n_qubits):
            ry = torch.tanh(z_2n[2 * q]) * np.pi
            rz = torch.tanh(z_2n[2 * q + 1]) * np.pi
            qml.RY(ry, wires=q)
            qml.RZ(rz, wires=q)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def qnode(z_2n, params_1d):
        n_params_per_layer = n_qubits * 3

        if use_reupload:
            for layer in range(n_layers):
                angle_encode_from_latent(z_2n)

                base = layer * n_params_per_layer
                for q in range(n_qubits):
                    p0 = base + q * 3
                    qml.RX(params_1d[p0], wires=q)
                    qml.RY(params_1d[p0 + 1], wires=q)
                    qml.RZ(params_1d[p0 + 2], wires=q)

                entangle(layer)
        else:
            angle_encode_from_latent(z_2n)

            for layer in range(n_layers):
                base = layer * n_params_per_layer
                for q in range(n_qubits):
                    p0 = base + q * 3
                    qml.RX(params_1d[p0], wires=q)
                    qml.RY(params_1d[p0 + 1], wires=q)
                    qml.RZ(params_1d[p0 + 2], wires=q)
                entangle(layer)

        # match your measurement design
        meas = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        meas += [qml.expval(qml.PauliY(i)) for i in range(min(2, n_qubits))]
        return meas

    return qnode

# -----------------------------
# Extract z latent (post-PCA) from the trained model for a given SMILES + target
# -----------------------------
@torch.no_grad()
def get_angle_latent_from_model(model, smiles: str, target_idx: int):
    qa_layer = model.attention.quantum_attention
    qc = qa_layer.quantum_circuit

    if qc.angle_pca_components is None or qc.angle_pca_mean is None:
        raise RuntimeError(
            "Angle PCA not loaded in the model. Check 'angle_pca_path' in config."
        )

    # mol -> graph encoder
    mol_feats, _ = model.graph_encoder([smiles])
    mol_feats = mol_feats.to(DEVICE)

    # query
    q = qa_layer.query_projection(mol_feats).squeeze(0)
    q = qa_layer.query_norm(q)

    # key from target embedding
    tid = torch.tensor([target_idx], device=DEVICE)
    emb = qa_layer.target_embeddings(tid).squeeze(0)
    key = qa_layer.key_projection(emb.unsqueeze(0)).squeeze(0)
    key = qa_layer.key_norm(key)

    combined = torch.cat([q, key], dim=-1).float()

    # --- ensure PCA tensors are on the same device as combined ---
    pca_mean = qc.angle_pca_mean.to(combined.device)
    pca_comp = qc.angle_pca_components.to(combined.device)

    x_centered = combined - pca_mean
    z = torch.matmul(pca_comp, x_centered)  # [2*n_qubits]
    return z


# -----------------------------
# Plot helpers
# -----------------------------
def plot_hist_params(params_1d, out_path, title):
    p = params_1d.detach().cpu().numpy().ravel()
    plt.figure(figsize=(7, 4))
    plt.hist(p, bins=40)
    plt.title(title)
    plt.xlabel("parameter value")
    plt.ylabel("count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")

def plot_hist_params_by_layer(params_1d, n_qubits, n_layers, out_path, title):
    p = params_1d.detach().cpu().numpy().ravel()
    per_layer = n_qubits * 3
    plt.figure(figsize=(10, 5))
    for L in range(n_layers):
        seg = p[L * per_layer:(L + 1) * per_layer]
        plt.hist(seg, bins=30, alpha=0.6, label=f"layer {L+1}")
    plt.title(title)
    plt.xlabel("parameter value")
    plt.ylabel("count")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")

def plot_grad_norms_by_layer(grad_1d, n_qubits, n_layers, out_path, title):
    g = grad_1d.detach().cpu().numpy().ravel()
    per_layer = n_qubits * 3
    norms = []
    for L in range(n_layers):
        seg = g[L * per_layer:(L + 1) * per_layer]
        norms.append(float(np.linalg.norm(seg, ord=2)))

    plt.figure(figsize=(7, 4))
    xs = np.arange(1, n_layers + 1)
    plt.plot(xs, norms, marker="o")
    plt.xticks(xs)
    plt.title(title)
    plt.xlabel("circuit layer")
    plt.ylabel("||∇θ_L score||₂ (single-input diagnostic)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")

    return norms

# -----------------------------
# Main
# -----------------------------
def main(
    smiles: str,
    target_name: str,
    target_idx: int,
    ckpt_path: str,
    out_dir: str = None,
):
    if ckpt_path is None:
        ckpt_path = find_latest_best_checkpoint(RESULTS_MODELS_DIR)

    ckpt = load_checkpoint(ckpt_path)
    print(f"✓ Using checkpoint: {ckpt_path}")

    fallback_config = os.path.join(PROJECT_DIR, "config_ui_run.json")
    if not os.path.exists(fallback_config):
        fallback_config = os.path.join(PROJECT_DIR, "config.json")

    tmp_cfg_path = make_temp_config_from_ckpt(ckpt, fallback_config)

    # build & load model
    model = MultiTargetPredictor(config_path=tmp_cfg_path, model_type="quantum").to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print("✓ Loaded checkpoint state_dict strictly (exact match).")

    qa_layer = model.attention.quantum_attention
    qc = qa_layer.quantum_circuit

    n_qubits = int(qc.n_qubits)
    n_layers = int(qc.n_layers)
    enc = str(qc.encoding_type)
    reup = bool(qc.use_data_reuploading)

    print(f"✓ Quantum settings: enc={enc}, reupload={reup}, q={n_qubits}, layers={n_layers}")
    print(f"✓ Target: {target_name} (target_idx={target_idx})")

    if out_dir is None:
        out_dir = os.path.join(PROJECT_DIR, "results", "plots", "training_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    # --- (1) Parameter distribution plots (no gradients needed)
    params = qa_layer.quantum_params[target_idx].detach().to(DEVICE).float()

    plot_hist_params(
        params,
        os.path.join(out_dir, f"quantum_params_hist_target{target_idx}.png"),
        title=f"Quantum parameters histogram (target_idx={target_idx}) | enc={enc}, reup={reup}, q={n_qubits}, L={n_layers}"
    )

    plot_hist_params_by_layer(
        params,
        n_qubits=n_qubits,
        n_layers=n_layers,
        out_path=os.path.join(out_dir, f"quantum_params_hist_by_layer_target{target_idx}.png"),
        title=f"Quantum parameters by layer (target_idx={target_idx}) | enc={enc}, reup={reup}, q={n_qubits}, L={n_layers}"
    )

    # --- (2) Gradient norm diagnostic (single input)
    # We do a SINGLE forward+backward on the scalar "attention score" computed from:
    #   score = attention_head( measurements( qnode(z, params) ) )
    # This is NOT training. It's a diagnostic snapshot of sensitivity/gradient flow.
    z = get_angle_latent_from_model(model, smiles=smiles, target_idx=target_idx).to(DEVICE).float()

    qnode = build_angle_qnode_lightning(n_qubits=n_qubits, n_layers=n_layers, use_reupload=reup)

    # make params trainable for gradient computation
    params_g = params.clone().detach().requires_grad_(True)

    # forward: circuit measurements
    meas = qnode(z, params_g)                # list of torch scalars
    meas_t = torch.stack(meas).float()       # [n_meas]

    # attention head on CPU too
    attention_head = qa_layer.attention_head.to(DEVICE).eval()
    score = attention_head(meas_t.unsqueeze(0)).squeeze()  # scalar

    # backward
    score.backward()
    grad = params_g.grad

    norms = plot_grad_norms_by_layer(
        grad,
        n_qubits=n_qubits,
        n_layers=n_layers,
        out_path=os.path.join(out_dir, f"grad_norms_by_layer_target{target_idx}.png"),
        title=f"Layer-wise gradient norms (single-input) | target_idx={target_idx}, enc={enc}, reup={reup}, q={n_qubits}, L={n_layers}"
    )

    # save meta
    meta = {
        "checkpoint": ckpt_path,
        "temp_config_used": tmp_cfg_path,
        "smiles": smiles,
        "target_name": target_name,
        "target_idx": int(target_idx),
        "encoding": enc,
        "reupload": reup,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "note": "This is a post-hoc diagnostic: one forward+backward pass on attention score using trained parameters. No retraining performed.",
        "grad_norms_per_layer": norms,
    }
    with open(os.path.join(out_dir, "diag_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved: {os.path.join(out_dir, 'diag_meta.json')}")
    print("✅ Done.")


if __name__ == "__main__":
    SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    TARGET_NAME = "CDK2"
    TARGET_IDX = 0

    CKPT_PATH = (
        r"C:\Users\Administrator\Desktop\Thesis_final\results\models"
        r"\quantum_gcn_enc-angle_q4_reup0_layers3_bs16_lr7e-04_ep20_20260106-175327_best.pt"
    )

    main(
        smiles=SMILES,
        target_name=TARGET_NAME,
        target_idx=TARGET_IDX,
        ckpt_path=CKPT_PATH,
        out_dir=None,
    )
