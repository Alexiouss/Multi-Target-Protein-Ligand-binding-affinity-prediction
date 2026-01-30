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

try:
    from multi_target_model_gcn_refactored import MultiTargetPredictor
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Could not import 'multi_target_model_gcn_refactored'.\n"
        f"Expected at: {os.path.join(SRC_DIR, 'multi_target_model_gcn_refactored.py')}\n"
    ) from e


# -----------------------------
# Utils
# -----------------------------
def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing 'model_state_dict'.")
    return ckpt


def make_temp_config_from_ckpt(ckpt: dict, fallback_config_path: str) -> str:
    """
    Instantiate the model with EXACT hyperparams matching the checkpoint.
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


@torch.no_grad()
def get_angle_latent_from_model(model, smiles: str, target_idx: int):
    """
    Produces z in POST-PCA space (length 2*n_qubits) exactly like your angle path.
    """
    qa_layer = model.attention.quantum_attention
    qc = qa_layer.quantum_circuit

    if qc.angle_pca_components is None or qc.angle_pca_mean is None:
        raise RuntimeError("Angle PCA not loaded (angle_pca_components/mean is None).")

    # 1) molecule graph encoder
    mol_feats, _ = model.graph_encoder([smiles])  # [1, 256] typically
    mol_feats = mol_feats.to(DEVICE)

    # 2) query
    q = qa_layer.query_projection(mol_feats).squeeze(0)
    q = qa_layer.query_norm(q)

    # 3) key (target embedding -> key projection)
    tid = torch.tensor([target_idx], device=DEVICE)
    emb = qa_layer.target_embeddings(tid).squeeze(0)
    key = qa_layer.key_projection(emb.unsqueeze(0)).squeeze(0)
    key = qa_layer.key_norm(key)

    combined = torch.cat([q, key], dim=-1).float()  # [2D]

    # 4) PCA -> latent
    x_centered = combined - qc.angle_pca_mean
    z = torch.matmul(qc.angle_pca_components, x_centered)  # [2*n_qubits]
    return z


# -----------------------------
# Quantum circuit for correlations
# -----------------------------
def build_angle_correlation_qnode(
    n_qubits: int,
    n_layers: int,
    use_reupload: bool,
    device_name: str = "lightning.qubit",
    shots: int | None = None,
):
    """
    QNode that returns:
      - singles: <Z_i>
      - pairs:   <Z_i Z_j> for all i<j
    """
    dev = qml.device(device_name, wires=n_qubits, shots=shots)

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

    @qml.qnode(dev, interface="torch")
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

        outs = []
        # singles
        for i in range(n_qubits):
            outs.append(qml.expval(qml.PauliZ(i)))
        # pairs
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                outs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
        return outs

    return qnode


def plot_heatmap(mat, title, out_png, xticks, yticks, vmin=-1.0, vmax=1.0):
    plt.figure(figsize=(5.5, 4.6))
    plt.imshow(mat, origin="upper", vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(label="value")
    plt.xticks(range(len(xticks)), xticks)
    plt.yticks(range(len(yticks)), yticks)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_png}")


def main(smiles, target_name, ckpt_path, target_idx=None, shots=None):
    ckpt = load_checkpoint(ckpt_path)
    print(f"✓ Using checkpoint: {ckpt_path}")

    fallback_config = os.path.join(PROJECT_DIR, "config_ui_run.json")
    if not os.path.exists(fallback_config):
        fallback_config = os.path.join(PROJECT_DIR, "config.json")

    tmp_cfg_path = make_temp_config_from_ckpt(ckpt, fallback_config)

    model = MultiTargetPredictor(config_path=tmp_cfg_path, model_type="quantum").to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print("✓ Loaded checkpoint state_dict strictly.")

    qa_layer = model.attention.quantum_attention
    qc = qa_layer.quantum_circuit

    n_qubits = int(qc.n_qubits)
    n_layers = int(qc.n_layers)
    enc = str(qc.encoding_type)
    reup = bool(qc.use_data_reuploading)

    print(f"✓ Quantum settings: enc={enc}, reupload={reup}, q={n_qubits}, layers={n_layers}")

    # resolve target_idx
    if target_idx is None:
        if target_name is not None and hasattr(model, "target_names") and target_name in model.target_names:
            target_idx = model.target_names.index(target_name)
        else:
            target_idx = 0
    print(f"✓ Using target_idx={target_idx}")

    params = qa_layer.quantum_params[target_idx].detach().to(DEVICE).float()
    z = get_angle_latent_from_model(model, smiles=smiles, target_idx=target_idx).to(DEVICE)

    # QNode (lightning.qubit)
    qnode = build_angle_correlation_qnode(
        n_qubits=n_qubits,
        n_layers=n_layers,
        use_reupload=reup,
        device_name="lightning.qubit",
        shots=shots,  # None = analytic expectations, int = shot-based estimate
    )

    with torch.no_grad():
        outs = qnode(z, params)
        outs = torch.stack(outs).float().cpu().numpy()

    # unpack
    singles = outs[:n_qubits]  # <Z_i>
    pair_vals = outs[n_qubits:]  # <Z_i Z_j> i<j

    # build matrices
    ZZ = np.eye(n_qubits, dtype=np.float32)
    C = np.eye(n_qubits, dtype=np.float32)

    idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            zz = float(pair_vals[idx])
            ZZ[i, j] = ZZ[j, i] = zz
            # connected correlation
            Cij = zz - float(singles[i]) * float(singles[j])
            C[i, j] = C[j, i] = Cij
            idx += 1

    # output dir
    out_dir = os.path.join(PROJECT_DIR, "results", "plots", "correlations")
    os.makedirs(out_dir, exist_ok=True)

    tag = f"enc-{enc}_q{n_qubits}_reup{int(reup)}_L{n_layers}_t{target_idx}_shots{shots if shots else 'analytic'}"
    out_zz = os.path.join(out_dir, f"zz_corr_{tag}.png")
    out_c  = os.path.join(out_dir, f"zz_connected_{tag}.png")

    ticks = [f"q{i}" for i in range(n_qubits)]
    plot_heatmap(
        ZZ,
        title=f"Raw Z–Z correlations ⟨Zi Zj⟩ ({tag})",
        out_png=out_zz,
        xticks=ticks,
        yticks=ticks,
        vmin=-1.0,
        vmax=1.0
    )
    plot_heatmap(
        C,
        title=f"Connected correlations Cij = ⟨ZiZj⟩−⟨Zi⟩⟨Zj⟩ ({tag})",
        out_png=out_c,
        xticks=ticks,
        yticks=ticks,
        vmin=-1.0,
        vmax=1.0
    )

    # save numbers for thesis
    meta = {
        "checkpoint": ckpt_path,
        "smiles": smiles,
        "target_name": target_name,
        "target_idx": int(target_idx),
        "encoding": enc,
        "reupload": bool(reup),
        "n_qubits": int(n_qubits),
        "n_layers": int(n_layers),
        "shots": shots,
        "singles_Z": singles.tolist(),
        "ZZ": ZZ.tolist(),
        "C_connected": C.tolist(),
    }
    out_json = os.path.join(out_dir, f"corr_values_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {out_json}")
    print("✅ Done.")


if __name__ == "__main__":
    SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    TARGET_NAME = "CDK2"
    CKPT_PATH = (
        r"C:\Users\Administrator\Desktop\Thesis_final\results\models"
        r"\quantum_gcn_enc-angle_q4_reup0_layers3_bs16_lr7e-04_ep20_20260106-175327_best.pt"
    )

    # shots=None gives analytic expectations (fast + clean).
    SHOTS = None

    main(
        smiles=SMILES,
        target_name=TARGET_NAME,
        ckpt_path=CKPT_PATH,
        target_idx=None,
        shots=SHOTS
    )
