import os
import sys
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pennylane as qml

# -----------------------------
# Paths / device (FIXED for scripts/)
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This script lives in: <project_root>/scripts/
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)  # <-- parent folder = project root

SRC_DIR = os.path.join(PROJECT_DIR, "src")
RESULTS_MODELS_DIR = os.path.join(PROJECT_DIR, "results", "models")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print("✓ PROJECT_DIR:", PROJECT_DIR)
print("✓ SRC_DIR:", SRC_DIR)
print("✓ RESULTS_MODELS_DIR:", RESULTS_MODELS_DIR)

# Import your model (now should work)
from multi_target_model_gcn_refactored import MultiTargetPredictor


# -----------------------------
# Utilities
# -----------------------------
def find_best_checkpoint_by_signature(
    models_dir: str,
    backbone: str,
    encoding: str,
    n_qubits: int,
    reupload: int,
) -> str:
    """
    Select exactly one '*_best.pt' checkpoint that matches the required signature.

    Priority:
      1) Match by filename if your new naming scheme is used.
      2) Otherwise, inspect checkpoint metadata (model_config) to match.

    Returns the most recently modified matching checkpoint.
    """
    backbone = backbone.lower()
    encoding = encoding.lower()
    reupload = int(reupload)
    n_qubits = int(n_qubits)

    cands = glob.glob(os.path.join(models_dir, "*_best.pt"))
    if not cands:
        raise FileNotFoundError(f"No '*_best.pt' checkpoints found in: {models_dir}")

    # ---------
    # (1) Fast path: match by filename (new naming scheme)
    # Example filename chunk:
    #   quantum_gcn_enc-angle_q4_reup0_..._best.pt
    # ---------
    token = f"_{backbone}_enc-{encoding}_q{n_qubits}_reup{reupload}_"
    by_name = [p for p in cands if token in os.path.basename(p)]
    if by_name:
        by_name.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return by_name[0]

    # ---------
    # (2) Fallback: open checkpoints and match by metadata
    # This supports older checkpoints.
    # ---------
    matches = []
    for p in cands:
        try:
            ckpt = torch.load(p, map_location="cpu")

            # backbone stored by your code in checkpoint
            ckpt_backbone = str(ckpt.get("backbone", "")).lower()

            # model_config may store encoding and n_qubits
            mc = ckpt.get("model_config", {}) or {}
            ckpt_enc = str(mc.get("encoding_type", "")).lower()
            ckpt_nq = int(mc.get("n_qubits", -1))

            # use_data_reuploading is in model_config in your setup
            # (if missing, assume 0)
            ckpt_reup = int(bool(mc.get("use_data_reuploading", False)))

            if (
                ckpt_backbone == backbone
                and ckpt_enc == encoding
                and ckpt_nq == n_qubits
                and ckpt_reup == reupload
            ):
                matches.append(p)
        except Exception:
            continue

    if not matches:
        raise FileNotFoundError(
            f"No matching checkpoint found for backbone={backbone}, "
            f"encoding={encoding}, n_qubits={n_qubits}, reupload={reupload} in {models_dir}"
        )

    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]



def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    return ckpt


def heatmap(Z, xs, ys, title, out_png, cbar_label):
    plt.figure(figsize=(7, 6))
    plt.imshow(
        Z,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        aspect="auto",
    )
    plt.colorbar(label=cbar_label)
    plt.xlabel(r"input slice dim 0")
    plt.ylabel(r"input slice dim 1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_png}")


# -----------------------------
# Build a plotting-only circuit that matches the ANGLE path
# Acceps a vector already in the "post-PCA space" (2*n_qubits).
#
#
# For visualization, we plot the circuit as a function of that (2*n_qubits) space
# because that is exactly what ends up feeding the RY/RZ angles.
# -----------------------------
def build_angle_circuit(n_qubits: int, n_layers: int, use_reupload: bool, device_name="default.qubit"):
    dev = qml.device(device_name, wires=n_qubits)

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
        # angle = tanh(feature)*pi then RY / RZ
        for q in range(n_qubits):
            ry = torch.tanh(z_2n[2*q]) * np.pi
            rz = torch.tanh(z_2n[2*q + 1]) * np.pi
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

        meas = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        meas += [qml.expval(qml.PauliY(i)) for i in range(min(2, n_qubits))]
        return meas

    return qnode


def compute_landscape(qnode, params, attention_head=None, grid=81, span=np.pi, out_mode="z0"):
    """
    out_mode:
      - "z0": plot <Z0>
      - "score": plot attention_head(measurements)
    """
    xs = np.linspace(-span, span, grid, dtype=np.float32)
    ys = np.linspace(-span, span, grid, dtype=np.float32)
    Z = np.zeros((grid, grid), dtype=np.float32)

    # base latent vector (post-PCA space)
    base = torch.zeros(params.shape[0]*0 + 1)  # dummy to satisfy lint
    # real base:
    base = torch.zeros(2 * N_QUBITS, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        for i, a0 in enumerate(xs):
            for j, a1 in enumerate(ys):
                z = base.clone()
                z[0] = float(a0)
                z[1] = float(a1)

                meas = qnode(z, params)                # list-like torch tensor
                meas_t = torch.stack(meas).float().to(next(attention_head.parameters()).device) if attention_head is not None else torch.stack(meas).float() # [n_meas]

                if out_mode == "z0":
                    val = meas_t[0].item()
                elif out_mode == "score":
                    if attention_head is None:
                        raise ValueError("attention_head is required for out_mode='score'")
                    val = attention_head(meas_t.unsqueeze(0)).squeeze().item()
                else:
                    raise ValueError("Unknown out_mode")

                Z[j, i] = float(val)

    return xs, ys, Z

import tempfile

def build_temp_config_from_checkpoint(ckpt, project_dir: str) -> str:
    """
    Creates a temporary config.json that matches the training run stored in checkpoint.
    Returns the path to the temp config file.
    """
    if "model_config" not in ckpt:
        raise KeyError("Checkpoint is missing 'model_config'. Cannot reconstruct exact model.")

    model_cfg = ckpt["model_config"]
    training_cfg = ckpt.get("training_config", {})

    # Targets: try dataset_meta first (most reliable), else fall back to existing config.json
    dataset_meta = ckpt.get("dataset_meta", {}) or {}
    target_names = dataset_meta.get("target_names", None)

    if target_names is None:
        # fallback: load current config.json only for target list
        fallback_cfg_path = os.path.join(project_dir, "config.json")
        if not os.path.exists(fallback_cfg_path):
            raise FileNotFoundError("No dataset_meta.target_names in checkpoint and no config.json to fall back on.")
        with open(fallback_cfg_path, "r", encoding="utf-8") as f:
            fallback_cfg = json.load(f)
        target_names = list(fallback_cfg.get("targets", {}).keys())

    # Build minimal config that your MultiTargetPredictor expects
    cfg = {
        "model": {
            **model_cfg,
            # ensure n_targets consistency
            "n_targets": int(model_cfg.get("n_targets", len(target_names))) if target_names else int(model_cfg.get("n_targets", 5)),
        },
        "targets": {t: t for t in target_names} if target_names else {},
        "training": training_cfg,
        "data": {
            "use_chembl": bool(dataset_meta.get("use_chembl", False)),
            "csv_path": dataset_meta.get("csv_path", None),
        },
    }

    # Write to a temp file
    fd, tmp_path = tempfile.mkstemp(prefix="ckpt_config_", suffix=".json", dir=os.path.join(project_dir, "scripts"))
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return tmp_path



# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ckpt_path = find_best_checkpoint_by_signature(
        RESULTS_MODELS_DIR,
        backbone="gcn",
        encoding="angle",
        n_qubits=4,
        reupload=0,
    )

    ckpt = load_checkpoint(ckpt_path)
    print(f"✓ Using checkpoint: {ckpt_path}")

    # ✅ Build model using EXACT model_config from checkpoint
    tmp_config_path = build_temp_config_from_checkpoint(ckpt, PROJECT_DIR)
    print(f"✓ Temp config created: {tmp_config_path}")

    model = MultiTargetPredictor(config_path=tmp_config_path, model_type="quantum").to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print("✓ Loaded checkpoint state_dict strictly (exact match).")


    # Pull settings from the loaded model
    qa_layer = model.attention.quantum_attention  # QuantumAttentionLayer
    qc_obj = qa_layer.quantum_circuit            # QuantumAttentionCircuit

    N_QUBITS = int(qc_obj.n_qubits)
    N_LAYERS = int(qc_obj.n_layers)

    print(f"✓ Model quantum settings: n_qubits={N_QUBITS}, n_layers={N_LAYERS}")

    # Choose a target to visualize
    target_idx = 0

    # Extract trained params for this target
    params = qa_layer.quantum_params[target_idx].detach().to(DEVICE).float()
    print(f"✓ Using target_idx={target_idx} params shape: {tuple(params.shape)}")

    # Extract attention_head (this makes the plot directly about your attention score)
    attention_head = qa_layer.attention_head.to(DEVICE).eval()

    # Build two plotting circuits (dense vs reupload)
    q_dense = build_angle_circuit(N_QUBITS, N_LAYERS, use_reupload=False, device_name="default.qubit")
    q_reup  = build_angle_circuit(N_QUBITS, N_LAYERS, use_reupload=True,  device_name="default.qubit")

    GRID = 81
    SPAN = np.pi

    # 1) Measurement landscapes: <Z0>
    xs, ys, Z1 = compute_landscape(q_dense, params, attention_head=None, grid=GRID, span=SPAN, out_mode="z0")
    heatmap(Z1, xs, ys,
            title="Angle Dense: measurement landscape (⟨Z0⟩)",
            out_png="landscape_angle_dense_Z0.png",
            cbar_label=r"$\langle Z_0 \rangle$")

    xs, ys, Z2 = compute_landscape(q_reup, params, attention_head=None, grid=GRID, span=SPAN, out_mode="z0")
    heatmap(Z2, xs, ys,
            title="Angle + Reupload: measurement landscape (⟨Z0⟩)",
            out_png="landscape_angle_reupload_Z0.png",
            cbar_label=r"$\langle Z_0 \rangle$")

    # 2) Score landscapes: attention_head(measurements) 
    xs, ys, S1 = compute_landscape(q_dense, params, attention_head=attention_head, grid=GRID, span=SPAN, out_mode="score")
    heatmap(S1, xs, ys,
            title="Angle Dense: attention score landscape",
            out_png="landscape_angle_dense_SCORE.png",
            cbar_label="attention score (pre-softmax)")

    xs, ys, S2 = compute_landscape(q_reup, params, attention_head=attention_head, grid=GRID, span=SPAN, out_mode="score")
    heatmap(S2, xs, ys,
            title="Angle + Reupload: attention score landscape",
            out_png="landscape_angle_reupload_SCORE.png",
            cbar_label="attention score (pre-softmax)")
