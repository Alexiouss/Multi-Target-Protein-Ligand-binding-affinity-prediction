import os
import math
import torch
import matplotlib.pyplot as plt

from multi_target_model_gcn_refactored import (
    MultiTargetPredictor,
    ClassicalAttentionLayer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# CHECKPOINTS
# -------------------------------------------------
quantum_ckpt = "results\models\quantum_gcn_enc-angle_q4_reup0_layers3_bs16_lr7e-04_ep20_20260106-175327_best.pt"
classical_ckpt = "results\models\classical_gcn_layers3_bs16_lr0p001_ep20_20260103-172425_best.pt"

# -------------------------------------------------
# TEST SMILES
# -------------------------------------------------
SMILES = [
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "CCO",
    "CC1=CC=C(C=C1)C(=O)C2=CC=CC=C2",
    "CN1CCN(CC1)C2=C(C=C3C(=C2)N=CN3C4=CC=CC=C4F)C#N",
]

import json
import tempfile

def load_trained_model(ckpt_path: str) -> MultiTargetPredictor:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model_type = ckpt["model_type_variant"]          # "quantum" or "classical"
    config_path = ckpt["config_path"]
    model_cfg = ckpt.get("model_config", {})         # snapshot of training config["model"]

    # Load the original config file
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Force the model section to match the checkpoint snapshot
    if "model" not in cfg:
        cfg["model"] = {}
    cfg["model"].update(model_cfg)

    # Write to a temporary config file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, tmp, indent=2)
    tmp_path = tmp.name
    tmp.close()

    # Build model using the patched temp config
    model = MultiTargetPredictor(
        config_path=tmp_path,
        model_type=model_type,
    ).to(DEVICE)

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model



# -------------------------------------------------
# CLASSICAL PRE-SOFTMAX LOGITS
# -------------------------------------------------
@torch.no_grad()
def classical_logits(att_layer: ClassicalAttentionLayer, mol_feats: torch.Tensor):
    B = mol_feats.shape[0]
    T = att_layer.n_targets

    # Q
    Q = att_layer.query_projection(mol_feats).unsqueeze(1)  # [B,1,D]

    # K
    target_idx = torch.arange(T, device=mol_feats.device)
    target_embs = att_layer.target_embeddings(target_idx)  # [T,D]
    K = target_embs.unsqueeze(0).expand(B, -1, -1)         # [B,T,D]

    mha = att_layer.attention
    D = mha.embed_dim
    H = mha.num_heads
    hd = D // H

    W = mha.in_proj_weight
    b = mha.in_proj_bias
    Wq, Wk = W[:D], W[D:2*D]
    bq, bk = b[:D], b[D:2*D]

    Qp = torch.nn.functional.linear(Q, Wq, bq)
    Kp = torch.nn.functional.linear(K, Wk, bk)

    Qh = Qp.view(B, 1, H, hd).transpose(1, 2)
    Kh = Kp.view(B, T, H, hd).transpose(1, 2)

    logits = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(hd)
    return logits.mean(dim=1).squeeze(1)  # [B,T]


# -------------------------------------------------
# ROW NORMALIZATION
# -------------------------------------------------
def row_zscore(S: torch.Tensor, eps=1e-8):
    mu = S.mean(dim=1, keepdim=True)
    sd = S.std(dim=1, keepdim=True).clamp_min(eps)
    return (S - mu) / sd


# -------------------------------------------------
# PLOTTING
# -------------------------------------------------
def plot_three(Sc, Sq, title, target_names=None):
    Sc = Sc.detach().cpu()
    Sq = Sq.detach().cpu()
    D = Sq - Sc

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(Sc, aspect="auto")
    axes[0].set_title("Classical logits (pre-softmax)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(Sq, aspect="auto")
    axes[1].set_title("Quantum logits (raw_scores)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(D, aspect="auto")
    axes[2].set_title("Quantum − Classical")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("Target")
        ax.set_ylabel("Sample")
        if target_names:
            ax.set_xticks(range(len(target_names)))
            ax.set_xticklabels(target_names, rotation=45, ha="right")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
@torch.no_grad()
def main():
    print("✓ Loading trained models...")
    q_model = load_trained_model(quantum_ckpt)
    c_model = load_trained_model(classical_ckpt)

    print("✓ Encoding molecules...")
    mol_feats, _ = q_model.graph_encoder(SMILES)
    mol_feats = mol_feats.to(DEVICE)

    print("✓ Quantum attention forward...")
    q_weights, q_attended, raw_scores = q_model.attention(mol_feats)
    Sq = raw_scores

    print("✓ Classical attention logits reconstruction...")
    Sc = classical_logits(c_model.attention, mol_feats)

    target_names = q_model.target_names

    # 1️⃣ Raw logits
    plot_three(
        Sc, Sq,
        title="Trained models – raw attention logits",
        target_names=target_names
    )

    # 2️⃣ Geometry view (row-normalized)
    plot_three(
        row_zscore(Sc),
        row_zscore(Sq),
        title="Trained models – row-normalized logits (geometry)",
        target_names=target_names
    )

    # 3️⃣ Attention weights (actual softmax outputs)
    c_weights, _ = c_model.attention(mol_feats)
    plot_three(
        c_weights,
        q_weights,
        title="Trained models – attention weights (softmax)",
        target_names=target_names
    )

    print("✓ Done")


if __name__ == "__main__":
    main()
