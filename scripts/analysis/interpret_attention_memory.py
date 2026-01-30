import re
import json
import importlib
from pathlib import Path
import tempfile
import copy

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os

from pathlib import Path

# --- Make src/ importable (same pattern as train.py) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)        # Thesis_final/
src_dir = os.path.join(project_dir, "src")


if src_dir not in sys.path:
    sys.path.append(src_dir)

os.chdir(project_dir)
os.environ["MPLBACKEND"] = "Agg"
from data_loader import create_data_loaders
from data_loader_chembl import create_data_loaders_from_chembl_csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = Path("results/models")
OUT_ROOT = Path("results/analysis/attention")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FNAME_RE = re.compile(
    r"^(?P<variant>classical|quantum)_(?P<backbone>gcn|gine)"
    r"(?:_enc-(?P<encoding>[a-zA-Z0-9]+))?"            # optional: _enc-angle
    r"(?:_q(?P<qubits>\d+))?"                          # optional: _q4
    r"(?:_reup(?P<reup>\d+))?"                         # optional: _reup0
    r"_layers(?P<layers>\d+)"
    r"_bs(?P<bs>\d+)"
    r"_lr(?P<lr>(?:\d+(?:p\d+)?|\d+(?:\.\d+)?)[eE][-+]?\d+|\d+p\d+|\d+(?:\.\d+)?)"
    r"_ep(?P<ep>\d+)"
    r"_(?P<ts>\d{8}-\d{6})"
    r"_(?P<tag>best|last)\.pt$"
)


def parse_model_name(filename: str) -> dict:
    m = FNAME_RE.match(filename)
    if not m:
        return {"raw": filename, "variant": "unknown", "backbone": "unknown", "bs": 32}
    d = m.groupdict()
    d["layers"] = int(d["layers"])
    d["bs"] = int(d["bs"])
    d["ep"] = int(d["ep"])
    lr_str = d["lr"].replace("p", ".")
    d["lr"] = float(lr_str)

    d["raw"] = filename
    return d

def _load_model_module(backbone: str):
    bb = backbone.lower()
    if bb == "gcn":
        module_name = "multi_target_model_gcn_refactored"
    elif bb == "gine":
        module_name = "multi_target_model_multiple_models_gine"
    else:
        raise ValueError(f"Unknown backbone '{backbone}' (expected gcn/gine)")
    module = importlib.import_module(module_name)
    if not hasattr(module, "MultiTargetPredictor"):
        raise AttributeError(f"Module '{module_name}' missing MultiTargetPredictor")
    return module

def make_model_init_config_for_checkpoint(fallback_config_path: str, ckpt: dict) -> str:
    """
    Create an effective config that matches:
      (1) dataset/targets from checkpoint
      (2) model hyperparams from checkpoint["model_config"] (CRITICAL for quantum)
    """
    with open(fallback_config_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    cfg = copy.deepcopy(base_cfg)

    # ---- 1) Dataset / targets ----
    ds = (ckpt.get("dataset_meta", {}) or {}) if isinstance(ckpt, dict) else {}
    target_names = ds.get("target_names", None)
    n_targets = ds.get("n_targets", None)

    if target_names is not None and n_targets is not None:
        cfg["targets"] = {t: t for t in target_names}
        cfg.setdefault("model", {})
        cfg["model"]["n_targets"] = int(n_targets)

    cfg.setdefault("data", {})
    cfg["data"]["use_chembl"] = bool(ds.get("use_chembl", cfg["data"].get("use_chembl", False)))
    if cfg["data"]["use_chembl"]:
        cfg["data"]["csv_path"] = ds.get("csv_path", cfg["data"].get("csv_path"))

    # ---- 2) Model hyperparams (THIS FIXES YOUR SHAPE MISMATCH) ----
    if isinstance(ckpt, dict):
        mcfg = ckpt.get("model_config", None)
        if isinstance(mcfg, dict):
            cfg.setdefault("model", {})
            cfg["model"].update(mcfg)

        # if you stored anything quantum-specific elsewhere, merge it too
        tcfg = ckpt.get("training_config", None)
        if isinstance(tcfg, dict) and "quantum" in tcfg and isinstance(tcfg["quantum"], dict):
            cfg.setdefault("quantum", {})
            cfg["quantum"].update(tcfg["quantum"])

    fd, tmp_path = tempfile.mkstemp(prefix="attn_modelcfg_", suffix=".json")
    Path(tmp_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return tmp_path


def make_test_loader_for_checkpoint(ckpt: dict, effective_config_path: str, num_workers: int, batch_size: int):
    """
    Returns (test_loader, is_chembl, csv_path_or_none).
    """
    ds = (ckpt.get("dataset_meta", {}) or {}) if isinstance(ckpt, dict) else {}

    # Priority 1: checkpoint
    use_chembl = ds.get("use_chembl", None)
    csv_path = ds.get("csv_path", None)

    # Priority 2: effective config
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
            raise ValueError("use_chembl=True but csv_path missing (checkpoint+config).")

        _, _, test_loader, _ = create_data_loaders_from_chembl_csv(
            csv_path=str(csv_path),
            batch_size=batch_size,
            num_workers=num_workers,
            use_chembl=True,
        )
        return test_loader, True, str(csv_path)

    # synthetic
    _, _, test_loader = create_data_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        config_path=effective_config_path,
    )
    return test_loader, False, None

@torch.no_grad()
def collect_attention_memory(model, loader, is_chembl: bool, max_batches: int | None):
    """
    Collect attention weights and summary stats.
    Faithful interpretation: attention = contribution weights of target embeddings
    to the shared attended context vector.
    """
    model.eval()
    eps = 1e-12

    attn_list = []
    entropy_list = []
    top1_list = []

    true_idx_list = []
    true_mass_list = []

    for b_i, batch in enumerate(tqdm(loader, desc="Collecting attention", leave=False)):
        if max_batches is not None and b_i >= max_batches:
            break

        preds, attn, _ = model(batch["molecules"])   # attn: [B,T]
        attn = attn.detach().float().cpu()

        attn_list.append(attn)

        # entropy per sample: -sum p log p
        ent = -(attn * (attn + eps).log()).sum(dim=1)
        entropy_list.append(ent)

        top1 = attn.argmax(dim=1)
        top1_list.append(top1)

        if is_chembl:
            if "target_indices" not in batch:
                raise KeyError("ChEMBL loader expected to provide batch['target_indices'].")
            true_idx = batch["target_indices"].detach().cpu().long()
            true_idx_list.append(true_idx)

            mass = attn[torch.arange(attn.shape[0]), true_idx]
            true_mass_list.append(mass)

    attn_all = torch.cat(attn_list, dim=0).numpy()
    entropy_all = torch.cat(entropy_list, dim=0).numpy()
    top1_all = torch.cat(top1_list, dim=0).numpy()

    out = {
        "attn": attn_all,
        "entropy": entropy_all,
        "top1": top1_all,
    }
    if is_chembl:
        out["true_target_idx"] = torch.cat(true_idx_list).numpy()
        out["true_target_mass"] = torch.cat(true_mass_list).numpy()

    return out

def plot_and_save(out_dir: Path, targets: list[str], stats: dict, is_chembl: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    attn = stats["attn"]      # [N,T]
    entropy = stats["entropy"]
    top1 = stats["top1"]

    # mean attention per target
    mean_attn = attn.mean(axis=0)
    plt.figure(figsize=(9, 4))
    plt.bar(targets, mean_attn)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean attention weight")
    plt.title("Target-memory contribution (mean attention per target)")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_attention_per_target.png", dpi=200)
    plt.close()

    # entropy distribution
    plt.figure(figsize=(6, 4))
    plt.hist(entropy, bins=60)
    plt.xlabel("Attention entropy")
    plt.ylabel("Count")
    plt.title("Attention entropy distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "attention_entropy_hist.png", dpi=200)
    plt.close()

    # top-1 frequency
    counts = np.bincount(top1, minlength=len(targets)).astype(np.float64)
    freq = counts / max(counts.sum(), 1.0)
    plt.figure(figsize=(9, 4))
    plt.bar(targets, freq)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Frequency")
    plt.title("Top-1 target embedding (most attended) frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "top1_target_frequency.png", dpi=200)
    plt.close()

    extra = {}
    if is_chembl and "true_target_mass" in stats:
        true_mass = stats["true_target_mass"]
        plt.figure(figsize=(6, 4))
        plt.hist(true_mass, bins=60)
        plt.xlabel("Attention weight on measured target")
        plt.ylabel("Count")
        plt.title("ChEMBL: attention mass on measured target index")
        plt.tight_layout()
        plt.savefig(out_dir / "chembl_true_target_mass_hist.png", dpi=200)
        plt.close()

        extra = {
            "true_target_mass_mean": float(true_mass.mean()),
            "true_target_mass_median": float(np.median(true_mass)),
        }

    summary = {
        "n_samples": int(attn.shape[0]),
        "n_targets": int(attn.shape[1]),
        "mean_attention": mean_attn.tolist(),
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(np.median(entropy)),
        "top1_frequency": freq.tolist(),
        **extra
    }
    (out_dir / "attention_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

def run(config_path: str, num_workers: int, max_batches: int | None):
    model_files = sorted(MODELS_DIR.glob("*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No checkpoints found in {MODELS_DIR}")

    for model_path in model_files:
        meta = parse_model_name(model_path.name)
        if meta["variant"] == "unknown":
            print(f"Skipping unrecognized filename: {model_path.name}")
            continue

        print(f"\n=== {model_path.name} ===")
        ckpt = torch.load(model_path, map_location=DEVICE)

        # make effective config that matches checkpoint dataset meta
        effective_cfg = make_model_init_config_for_checkpoint(config_path, ckpt if isinstance(ckpt, dict) else {})


        # make loader matching dataset
        test_loader, is_chembl, csv_path = make_test_loader_for_checkpoint(
            ckpt=ckpt if isinstance(ckpt, dict) else {},
            effective_config_path=effective_cfg,
            num_workers=num_workers,
            batch_size=meta["bs"],
        )

        module = _load_model_module(meta["backbone"])
        ModelClass = module.MultiTargetPredictor

        # checkpoint formats
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            variant = ckpt.get("model_type_variant", meta["variant"])
            backbone = ckpt.get("backbone", meta["backbone"])
        else:
            state = ckpt
            variant = meta["variant"]
            backbone = meta["backbone"]

        model = ModelClass(config_path=effective_cfg, model_type=variant).to(DEVICE)
        model.load_state_dict(state, strict=True)

        targets = list(model.target_names)

        stats = collect_attention_memory(model, test_loader, is_chembl=is_chembl, max_batches=max_batches)

        out_dir = OUT_ROOT / model_path.stem
        plot_and_save(out_dir, targets, stats, is_chembl=is_chembl)

        # save raw arrays
        np.savez_compressed(
            out_dir / "attention_raw.npz",
            attn=stats["attn"],
            entropy=stats["entropy"],
            top1=stats["top1"],
            true_target_idx=stats.get("true_target_idx", None),
            true_target_mass=stats.get("true_target_mass", None),
        )

        # small run metadata
        run_meta = {
            "checkpoint": str(model_path),
            "variant": variant,
            "backbone": backbone,
            "batch_size": meta["bs"],
            "is_chembl": bool(is_chembl),
            "csv_path": csv_path,
            "max_batches_used": max_batches,
            "interpretation_note": (
                "Attention is interpreted as target-memory contribution weights to a shared context vector "
                "(not per-target causal attribution)."
            ),
        }
        (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

        print(f"Saved → {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_ui_run.json")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=-1, help="Use -1 for all.")
    args = ap.parse_args()

    max_batches = None if args.max_batches < 0 else int(args.max_batches)
    run(config_path=args.config, num_workers=args.num_workers, max_batches=max_batches)
