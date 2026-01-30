"""

Visualization #4: Measurement statistics (bitstring probabilities) using Pennylane lightning.qubit
- Loads a specific checkpoint
- Reconstructs the exact model config from the checkpoint (so state_dict matches)
- Extracts the POST-PCA latent z for a given SMILES + target
- Builds a shot-based QNode on lightning.qubit returning qml.counts()
- Saves:
    - bitstring_probs_*.png
    - bitstring_counts_*.json
    - bitstring_meta_*.json

"""

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
    We do this by loading the base config and overwriting relevant model fields from ckpt["model_config"] (if present).
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
            "n_qubits",
            "n_layers",
            "feature_dim",
            "encoding_type",
            "use_data_reuploading",
            "pca_path",
            "angle_pca_path",
            "n_targets",
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
# Latent vector extraction (post-PCA) from the real model
# -----------------------------
@torch.no_grad()
def get_angle_latent_from_model(model, smiles: str, target_idx: int):
    """
    Produces z in the POST-PCA space (length = 2*n_qubits) exactly like your angle path does.

    Steps:
      SMILES -> graph_encoder -> query_projection + norm = query (D)
      target_embedding -> key_projection + norm = key (D)
      combined = [query||key] (2D)
      z = angle_pca_components @ (combined - mean)   (2*n_qubits)
    """
    qa_layer = model.attention.quantum_attention
    qc = qa_layer.quantum_circuit

    if qc.angle_pca_components is None or qc.angle_pca_mean is None:
        raise RuntimeError(
            "Angle PCA not loaded in the model (angle_pca_components/mean is None). "
            "Make sure your model config points to the right 'angle_pca_path' and the file exists."
        )

    # 1) molecule features
    mol_feats, _ = model.graph_encoder([smiles])
    mol_feats = mol_feats.to(DEVICE)  # [1, 256] typically

    # 2) query
    q = qa_layer.query_projection(mol_feats).squeeze(0)  # [D]
    q = qa_layer.query_norm(q)

    # 3) key for target
    tid = torch.tensor([target_idx], device=DEVICE)
    emb = qa_layer.target_embeddings(tid).squeeze(0)  # [D]
    key = qa_layer.key_projection(emb.unsqueeze(0)).squeeze(0)
    key = qa_layer.key_norm(key)

    combined = torch.cat([q, key], dim=-1).float()  # [2D]

    # 4) PCA to latent space
    x_centered = combined - qc.angle_pca_mean
    z = torch.matmul(qc.angle_pca_components, x_centered)  # [2*n_qubits]
    return z


# -----------------------------
# Shot-based circuit on lightning.qubit
# -----------------------------
def build_angle_sampling_qnode_lightning(
    n_qubits: int,
    n_layers: int,
    use_reupload: bool,
    shots: int = 5000,
):
    """
    Same structure as your angle circuit, returns qml.counts() for bitstring probabilities.
    Uses lightning.qubit as requested.
    """
    # IMPORTANT: lightning.qubit is CPU-based; we keep tensors on CPU for qnode calls.
    dev = qml.device("lightning.qubit", wires=n_qubits, shots=shots)

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

    @qml.qnode(dev, interface="torch")  # sampling => no adjoint
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

        return qml.counts(wires=range(n_qubits))

    return qnode


def counts_to_probs(counts: dict, shots: int, n_qubits: int):
    """
    Normalize counts to probabilities and include all bitstrings.
    Keys can be '0101' or tuples depending on device; normalize to strings.
    """
    probs = {format(i, f"0{n_qubits}b"): 0.0 for i in range(2**n_qubits)}

    for k, v in counts.items():
        if isinstance(k, tuple):
            k = "".join(map(str, k))
        probs[str(k)] = float(v) / float(shots)

    keys = sorted(probs.keys())
    vals = [probs[k] for k in keys]
    return keys, vals


def plot_bitstring_probs(keys, vals, title, out_png):
    plt.figure(figsize=(10, 4))
    plt.bar(keys, vals)
    plt.xlabel("bitstring")
    plt.ylabel("P(bitstring)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_png}")


def main(
    smiles: str,
    target_name: str = None,
    target_idx: int = None,
    ckpt_path: str = None,
    shots: int = 5000,
    out_dir: str = None,
):
    # checkpoint
    if ckpt_path is None:
        ckpt_path = find_latest_best_checkpoint(RESULTS_MODELS_DIR)
    ckpt = load_checkpoint(ckpt_path)
    print(f"✓ Using checkpoint: {ckpt_path}")

    # fallback config
    fallback_config = os.path.join(PROJECT_DIR, "config_ui_run.json")
    if not os.path.exists(fallback_config):
        fallback_config = os.path.join(PROJECT_DIR, "config.json")

    tmp_cfg_path = make_temp_config_from_ckpt(ckpt, fallback_config)

    # model
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

    if enc != "angle":
        raise ValueError(f"This script expects encoding_type='angle'. Got enc='{enc}'")

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

    # params
    params = qa_layer.quantum_params[target_idx].detach().to(DEVICE).float()
    print(f"✓ params shape: {tuple(params.shape)}")

    # latent z (post-PCA)
    z = get_angle_latent_from_model(model, smiles=smiles, target_idx=target_idx).to(DEVICE).float()
    print(f"✓ latent z shape: {tuple(z.shape)}")

    # output dir
    if out_dir is None:
        out_dir = os.path.join(PROJECT_DIR, "results", "plots", "bitstrings")
    os.makedirs(out_dir, exist_ok=True)

    # build lightning sampling qnode
    try:
        q_sample = build_angle_sampling_qnode_lightning(
            n_qubits=n_qubits,
            n_layers=n_layers,
            use_reupload=reup,
            shots=shots,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to create 'lightning.qubit' device.\n"
            "Make sure you have PennyLane-Lightning installed (pennylane-lightning).\n"
            f"Original error: {e}"
        ) from e

    # lightning.qubit runs on CPU => move to CPU explicitly
    z_cpu = z.detach().cpu()
    params_cpu = params.detach().cpu()

    counts = q_sample(z_cpu, params_cpu)  # dict
    keys, vals = counts_to_probs(counts, shots=shots, n_qubits=n_qubits)

    # save plot
    out_png = os.path.join(out_dir, f"bitstring_probs_target{target_idx}_shots{shots}.png")
    plot_bitstring_probs(
        keys,
        vals,
        title=f"P(z) bitstring distribution | enc={enc}, reupload={reup}, q={n_qubits}, layers={n_layers}, target_idx={target_idx}, shots={shots}",
        out_png=out_png,
    )

    # save raw counts + meta
    counts_path = os.path.join(out_dir, f"bitstring_counts_target{target_idx}_shots{shots}.json")
    with open(counts_path, "w", encoding="utf-8") as f:
        # counts keys may be tuples; normalize keys to str for JSON
        norm_counts = {}
        for k, v in counts.items():
            if isinstance(k, tuple):
                k = "".join(map(str, k))
            norm_counts[str(k)] = int(v)
        json.dump(norm_counts, f, indent=2)

    meta_path = os.path.join(out_dir, f"bitstring_meta_target{target_idx}_shots{shots}.json")
    meta = {
        "checkpoint": ckpt_path,
        "tmp_config": tmp_cfg_path,
        "smiles": smiles,
        "target_name": target_name,
        "target_idx": int(target_idx),
        "encoding": enc,
        "reupload": bool(reup),
        "n_qubits": int(n_qubits),
        "n_layers": int(n_layers),
        "shots": int(shots),
        "device": "lightning.qubit",
        "output_plot": out_png,
        "output_counts": counts_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved: {counts_path}")
    print(f"✓ Saved: {meta_path}")
    print("✅ Done.")


if __name__ == "__main__":
    SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # molecule
    TARGET_NAME = "CDK2"
    TARGET_IDX = None

    CKPT_PATH = (
        r"C:\Users\Administrator\Desktop\Thesis_final\results\models"
        r"\quantum_gcn_enc-angle_q4_reup0_layers3_bs16_lr7e-04_ep20_20260106-175327_best.pt"
    )

    SHOTS = 5000
    OUT_DIR = None

    main(
        smiles=SMILES,
        target_name=TARGET_NAME,
        target_idx=TARGET_IDX,
        ckpt_path=CKPT_PATH,
        shots=SHOTS,
        out_dir=OUT_DIR,
    )
