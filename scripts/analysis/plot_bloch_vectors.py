import os
import sys
import glob
import json
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt
import pennylane as qml

# -----------------------------
# Settings
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Import your model (from src/)
# -----------------------------
try:
    from multi_target_model_gcn_refactored import MultiTargetPredictor
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Could not import 'multi_target_model_gcn_refactored'.\n"
        "Make sure:\n"
        f"  - file exists at: {os.path.join(SRC_DIR, 'multi_target_model_gcn_refactored.py')}\n"
        "  - and that you run this script from scripts/ (it uses PROJECT_DIR/src on sys.path)\n"
    ) from e

# -----------------------------
# Stage colors + legend labels  ✅ NEW
# -----------------------------
STAGE_COLORS = [
    "red",     # stage 0: after encoding
    "blue",    # stage 1: after layer 1
    "green",   # stage 2: after layer 2
    "purple",  # stage 3: after layer 3
]
STAGE_LABELS = [
    "after encoding",
    "after layer 1",
    "after layer 2",
    "after layer 3",
]

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
    Ensure we instantiate a model with the SAME quantum hyperparams as the checkpoint.
    We do this by loading the config and overwriting model fields if they exist in ckpt.
    """
    if ckpt.get("config_path") and os.path.exists(ckpt["config_path"]):
        base_config_path = ckpt["config_path"]
    else:
        base_config_path = fallback_config_path

    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("model", {})

    # If the ckpt saved model_config/training_config
    ckpt_model_cfg = ckpt.get("model_config", None)
    if isinstance(ckpt_model_cfg, dict):
        # overwrite only relevant fields if present
        for k in ["n_qubits", "n_layers", "feature_dim", "encoding_type", "use_data_reuploading",
                  "pca_path", "angle_pca_path", "n_targets"]:
            if k in ckpt_model_cfg:
                model_cfg[k] = ckpt_model_cfg[k]

        cfg["model"] = model_cfg

    # write to temp config
    fd, tmp_path = tempfile.mkstemp(prefix="ckpt_config_", suffix=".json", dir=os.path.dirname(__file__))
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"✓ Temp config created: {tmp_path}")
    return tmp_path


# -----------------------------
# Build "partial circuit" qnodes for Bloch vectors
# -----------------------------
def build_angle_partial_qnode(
    n_qubits: int,
    n_layers: int,
    use_reupload: bool,
    upto_layer: int,   # 0 = after encoding only, 1 = after layer 1, ...
    device_name: str = "lightning.qubit",
):
    """
    Returns a QNode that:
      - encodes a latent vector z (length 2*n_qubits) into RY/RZ
      - optionally reuploads each layer (if use_reupload)
      - applies variational layers up to upto_layer
      - returns <X>,<Y>,<Z> for each qubit (Bloch vector components)
    """
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
        # matches your tanh(feature)*pi then RY/RZ
        for q in range(n_qubits):
            ry = torch.tanh(z_2n[2 * q]) * np.pi
            rz = torch.tanh(z_2n[2 * q + 1]) * np.pi
            qml.RY(ry, wires=q)
            qml.RZ(rz, wires=q)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def qnode(z_2n, params_1d):
        n_params_per_layer = n_qubits * 3

        if use_reupload:
            # re-encode before each layer
            for layer in range(min(upto_layer, n_layers)):
                angle_encode_from_latent(z_2n)
                base = layer * n_params_per_layer
                for q in range(n_qubits):
                    p0 = base + q * 3
                    qml.RX(params_1d[p0], wires=q)
                    qml.RY(params_1d[p0 + 1], wires=q)
                    qml.RZ(params_1d[p0 + 2], wires=q)
                entangle(layer)

            if upto_layer == 0:
                angle_encode_from_latent(z_2n)

        else:
            # encode once
            angle_encode_from_latent(z_2n)

            # apply first upto_layer layers
            for layer in range(min(upto_layer, n_layers)):
                base = layer * n_params_per_layer
                for q in range(n_qubits):
                    p0 = base + q * 3
                    qml.RX(params_1d[p0], wires=q)
                    qml.RY(params_1d[p0 + 1], wires=q)
                    qml.RZ(params_1d[p0 + 2], wires=q)
                entangle(layer)

        # Bloch components for each qubit
        outs = []
        for q in range(n_qubits):
            outs.append(qml.expval(qml.PauliX(q)))
            outs.append(qml.expval(qml.PauliY(q)))
            outs.append(qml.expval(qml.PauliZ(q)))
        return outs

    return qnode


# -----------------------------
# Latent vector extraction from the real model
# -----------------------------
@torch.no_grad()
def get_angle_latent_from_model(model, smiles: str, target_idx: int):
    """
    Produces z in the POST-PCA space (length = 2*n_qubits) exactly like your angle path does.
    """
    qa_layer = model.attention.quantum_attention
    qc = qa_layer.quantum_circuit

    if qc.angle_pca_components is None or qc.angle_pca_mean is None:
        raise RuntimeError(
            "Angle PCA not loaded in the model (angle_pca_components/mean is None). "
            "Make sure your model config points to the right 'angle_pca_path' and the file exists."
        )

    mol_feats, _ = model.graph_encoder([smiles])
    mol_feats = mol_feats.to(DEVICE)

    q = qa_layer.query_projection(mol_feats).squeeze(0)
    q = qa_layer.query_norm(q)

    tid = torch.tensor([target_idx], device=DEVICE)
    emb = qa_layer.target_embeddings(tid).squeeze(0)
    key = qa_layer.key_projection(emb.unsqueeze(0)).squeeze(0)
    key = qa_layer.key_norm(key)

    combined = torch.cat([q, key], dim=-1).float()
    x_centered = combined - qc.angle_pca_mean
    z = torch.matmul(qc.angle_pca_components, x_centered)
    return z


# -----------------------------
# Plotting (Bloch sphere + colored stage dots)
# -----------------------------
def plot_bloch_sphere_with_trajectory(ax, traj_xyz, title, stage_colors, stage_labels, add_legend=False):
    """
    Draw a Bloch sphere wireframe + trajectory line + colored dots per stage.
    """
    # sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.25)

    # axes
    ax.plot([0, 1], [0, 0], [0, 0], linewidth=1, alpha=0.4)
    ax.plot([0, 0], [0, 1], [0, 0], linewidth=1, alpha=0.4)
    ax.plot([0, 0], [0, 0], [0, 1], linewidth=1, alpha=0.4)

    traj_xyz = np.asarray(traj_xyz)

    # trajectory line (subtle)
    ax.plot(
        traj_xyz[:, 0], traj_xyz[:, 1], traj_xyz[:, 2],
        color="gray", alpha=0.35, linewidth=1.5
    )

    # colored dots per stage
    n_stages = traj_xyz.shape[0]
    for s in range(n_stages):
        c = stage_colors[s] if s < len(stage_colors) else "black"
        lab = stage_labels[s] if (add_legend and s < len(stage_labels)) else None
        ax.scatter(
            traj_xyz[s, 0], traj_xyz[s, 1], traj_xyz[s, 2],
            color=c, s=60, depthshade=True, label=lab
        )

    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel("⟨X⟩"); ax.set_ylabel("⟨Y⟩"); ax.set_zlabel("⟨Z⟩")
    ax.set_title(title)


def main(
    smiles: str = None,
    target_name: str = None,
    target_idx: int = None,
    ckpt_path: str = None,
    out_dir: str = None,
):
    # checkpoint
    if ckpt_path is None:
        ckpt_path = find_latest_best_checkpoint(RESULTS_MODELS_DIR)
    ckpt = load_checkpoint(ckpt_path)
    print(f"✓ Using checkpoint: {ckpt_path}")

    # config fallback
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

    print(f"✓ Quantum settings: encoding={enc}, reupload={reup}, n_qubits={n_qubits}, n_layers={n_layers}")

    # resolve target index
    if target_idx is None:
        if target_name is not None:
            if hasattr(model, "target_names") and target_name in model.target_names:
                target_idx = model.target_names.index(target_name)
            else:
                raise ValueError(f"target_name='{target_name}' not found in model.target_names.")
        else:
            target_idx = 0
    print(f"✓ Using target_idx={target_idx}")

    # params for this target
    params = qa_layer.quantum_params[target_idx].detach().to(DEVICE).float()
    print(f"✓ params shape: {tuple(params.shape)}")

    # latent z
    if smiles is not None:
        try:
            z = get_angle_latent_from_model(model, smiles=smiles, target_idx=target_idx).to(DEVICE)
            print(f"✓ Using latent z from SMILES='{smiles}' with shape {tuple(z.shape)}")
        except Exception as e:
            print(f"⚠️ Could not extract latent from SMILES due to: {e}")
            print("   Falling back to random latent.")
            z = torch.randn(2 * n_qubits, dtype=torch.float32, device=DEVICE)
    else:
        print("⚠️ No SMILES provided. Using random latent vector z.")
        z = torch.randn(2 * n_qubits, dtype=torch.float32, device=DEVICE)

    # output dir
    if out_dir is None:
        out_dir = os.path.join(PROJECT_DIR, "results", "plots", "bloch_vectors")
    os.makedirs(out_dir, exist_ok=True)

    # stages: 0..n_layers
    stages = list(range(0, n_layers + 1))

    # Trim color/label lists to match number of stages
    stage_colors = STAGE_COLORS[: len(stages)]
    stage_labels = STAGE_LABELS[: len(stages)]

    # Build qnodes for each stage
    qnodes = [
        build_angle_partial_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            use_reupload=reup,
            upto_layer=k,
            device_name="default.qubit"
        )
        for k in stages
    ]

    # Evaluate Bloch vectors
    bloch = []
    with torch.no_grad():
        for k, qn in zip(stages, qnodes):
            outs = qn(z, params)
            v = torch.stack(outs).float().cpu().numpy()
            v = v.reshape(n_qubits, 3)
            bloch.append(v)

    bloch = np.asarray(bloch)  # [n_stages, n_qubits, 3]

    # --------------- Plot 3D Bloch spheres (one per qubit) ---------------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(4 * n_qubits, 4))
    axes = []
    for q in range(n_qubits):
        ax = fig.add_subplot(1, n_qubits, q + 1, projection="3d")
        axes.append(ax)
        traj = bloch[:, q, :]

        # add legend only on first subplot
        plot_bloch_sphere_with_trajectory(
            ax,
            traj_xyz=traj,
            title=f"Qubit {q}",
            stage_colors=stage_colors,
            stage_labels=stage_labels,
            add_legend=(q == 0),
        )

    handles, labels = axes[0].get_legend_handles_labels()

    # Put legend ABOVE the title (no overlap)
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, 1.12)  # pushes legend higher
        )

    # Title slightly lower than legend
    fig.suptitle(
        f"Bloch-vector trajectories across circuit depth\n"
        f"(encoding={enc}, reupload={reup}, target_idx={target_idx})",
        y=1.03
    )

    # Leave top margin for legend+title
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out1 = os.path.join(out_dir, "bloch_trajectories.png")
    plt.savefig(out1, dpi=300, bbox_inches="tight")

    print(f"✓ Saved: {out1}")

    # --------------- Plot vector length (purity proxy) vs depth ---------------
    lengths = np.linalg.norm(bloch, axis=-1)  # [n_stages, n_qubits]

    plt.figure(figsize=(7, 4))
    for q in range(n_qubits):
        plt.plot(stages, lengths[:, q], marker="o", label=f"qubit {q}")
    plt.xlabel("Stage (0=after encoding, k=after k layers)")
    plt.ylabel("Bloch vector length ||r||")
    plt.title("Single-qubit purity proxy across depth (||r||)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(out_dir, "bloch_lengths.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out2}")

    # --------------- Save raw values for reproducibility ---------------
    npy_path = os.path.join(out_dir, "bloch_vectors.npy")
    np.save(npy_path, bloch)
    meta = {
        "checkpoint": ckpt_path,
        "encoding": enc,
        "reupload": reup,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "target_idx": int(target_idx),
        "smiles": smiles,
        "stages": stages,
    }
    with open(os.path.join(out_dir, "bloch_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved: {npy_path}")
    print(f"✓ Saved: {os.path.join(out_dir, 'bloch_meta.json')}")
    print("✅ Done.")


if __name__ == "__main__":
    SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    TARGET_NAME = "CDK2"
    TARGET_IDX = None
    CKPT_PATH = (
        r"C:\Users\Administrator\Desktop\Thesis_final\results\models"
        r"\quantum_gcn_enc-angle_q4_reup0_layers3_bs16_lr7e-04_ep20_20260106-175327_best.pt"
    )
    OUT_DIR = None

    main(
        smiles=SMILES,
        target_name=TARGET_NAME,
        target_idx=TARGET_IDX,
        ckpt_path=CKPT_PATH,
        out_dir=OUT_DIR,
    )
