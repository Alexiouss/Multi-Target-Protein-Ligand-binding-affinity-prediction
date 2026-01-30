import re
import json
import importlib
from pathlib import Path
import tempfile
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lifelines.utils import concordance_index

from data_loader import create_data_loaders
from data_loader_chembl import create_data_loaders_from_chembl_csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ You want results/plots/...
MODELS_DIR  = Path("results/models")
RESULTS_DIR = Path("results/plots/test_results")
PLOTS_DIR   = Path("results/plots/test_curves")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FNAME_RE = re.compile(
    r"^(?P<variant>classical|quantum)_(?P<backbone>gcn|gine)"
    r"(?:_enc-(?P<encoding>[a-zA-Z0-9]+))?"          # optional: _enc-angle
    r"(?:_q(?P<qubits>\d+))?"                         # optional: _q4
    r"(?:_reup(?P<reup>\d+))?"                        # optional: _reup0
    r"_layers(?P<layers>\d+)"
    r"_bs(?P<bs>\d+)"
    r"_lr(?P<lr>(?:\d+(?:p\d+)?|\d+(?:\.\d+)?)[eE][-+]?\d+|\d+p\d+|\d+(?:\.\d+)?)"
    r"_ep(?P<ep>\d+)"
    r"_(?P<ts>\d{8}-\d{6})"
    r"_(?P<tag>best|last)\.pt$"
)

from datetime import datetime

def json_safe(obj):
    """Make objects JSON serializable (Path, numpy, torch, etc.)."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj

def debug_effective_config(effective_config_path: str, ckpt_obj: dict, meta: dict):
    with open(effective_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    targets_cfg = list(cfg.get("targets", {}).keys())

    print("\n================ CONFIG CHECK ================")
    print("MODEL FILE:", meta.get("raw"))
    print("EFFECTIVE CONFIG:", effective_config_path)

    if isinstance(ckpt_obj, dict):
        print("CKPT backbone/variant:", ckpt_obj.get("backbone"), ckpt_obj.get("model_type_variant"))
        print("CKPT dataset_meta:", ckpt_obj.get("dataset_meta", {}))
        print("CKPT model_config keys:", list((ckpt_obj.get("model_config") or {}).keys()))

    print("CFG data.use_chembl:", data_cfg.get("use_chembl"))
    print("CFG data.csv_path:", data_cfg.get("csv_path"))
    print("CFG model.n_targets:", model_cfg.get("n_targets"))
    print("CFG targets len:", len(targets_cfg))
    print("CFG first targets:", targets_cfg[:5])

    # quantum-related (may exist for classical too)
    print("CFG encoding_type:", model_cfg.get("encoding_type"))
    print("CFG n_qubits:", model_cfg.get("n_qubits"))
    print("CFG pca_path:", model_cfg.get("pca_path"))
    print("CFG angle_pca_path:", model_cfg.get("angle_pca_path"))
    print("================================================\n")


def write_all_results(all_results: list, out_dir: Path, basename: str = "all_models_results"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pretty JSON (single file)
    json_path = out_dir / f"{basename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=json_safe)

    # JSON Lines (optional, convenient for pandas)
    jsonl_path = out_dir / f"{basename}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, default=json_safe) + "\n")

    return str(json_path), str(jsonl_path)
def parse_lr(lr_str: str) -> float:
    # supports: 0p001, 0.001, 7e-04, 1E-3
    s = lr_str.replace("p", ".")
    return float(s)

def parse_model_name(filename: str) -> dict:
    m = FNAME_RE.match(filename)
    if not m:
        return {"raw": filename, "variant": "unknown", "backbone": "unknown", "bs": 32}

    d = m.groupdict()
    d["layers"] = int(d["layers"])
    d["bs"] = int(d["bs"])
    d["ep"] = int(d["ep"])
    d["lr"] = parse_lr(d["lr"])
    d["raw"] = filename

    # optional fields
    if d.get("qubits") is not None:
        d["qubits"] = int(d["qubits"])
    if d.get("reup") is not None:
        d["reup"] = int(d["reup"])

    # if not present, set friendly defaults
    d["encoding"] = d.get("encoding") or None
    d["qubits"] = d.get("qubits") if "qubits" in d else None
    d["reup"] = d.get("reup") if "reup" in d else None

    return d


def make_model_init_config_for_checkpoint(fallback_config_path: str, ckpt: dict) -> str:
    """
    Create a temp config that matches BOTH:
      - targets/n_targets
      - quantum/model hyperparams needed to reconstruct exact module shapes
    """
    with open(fallback_config_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    cfg = copy.deepcopy(base_cfg)

    # ---- 1) Targets / dataset (same as before) ----
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

    # ---- 2) Model hyperparams from checkpoint ----
    # Newer ckpts: ckpt["model_config"] should include quantum settings & dims
    if isinstance(ckpt, dict):
        mcfg = ckpt.get("model_config", None)
        if isinstance(mcfg, dict):
            # merge into cfg["model"] (or wherever your code reads from)
            cfg.setdefault("model", {})
            # shallow merge is usually enough; if you have nested dicts, merge those too
            for k, v in mcfg.items():
                cfg["model"][k] = v

        # Sometimes quantum settings are stored in training_config or under different keys
        tcfg = ckpt.get("training_config", None)
        if isinstance(tcfg, dict):
            # If your model reads quantum params from cfg["quantum"], merge that too
            if "quantum" in tcfg and isinstance(tcfg["quantum"], dict):
                cfg.setdefault("quantum", {})
                cfg["quantum"].update(tcfg["quantum"])

    fd, tmp_path = tempfile.mkstemp(prefix="eval_modelcfg_", suffix=".json")
    Path(tmp_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return tmp_path


def load_target_names(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return list(cfg.get("targets", {}).keys())

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

def make_test_loader_for_checkpoint(ckpt: dict, fallback_config_path: str, num_workers: int, batch_size: int):
    """
    Builds the correct test loader based on the dataset the model was trained on.
    Priority:
      1) ckpt["dataset_meta"] if present
      2) fallback_config_path (config_ui_run.json)
    """
    # --- read dataset info ---
    ds = {}
    if isinstance(ckpt, dict):
        ds = ckpt.get("dataset_meta", {}) or {}

    use_chembl = ds.get("use_chembl", None)
    csv_path = ds.get("csv_path", None)

    # fallback to config if checkpoint doesn't contain dataset info (old models)
    if use_chembl is None:
        with open(fallback_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        data_cfg = cfg.get("data", {})
        use_chembl = bool(data_cfg.get("use_chembl", False))
        csv_path = data_cfg.get("csv_path", None)

    if use_chembl:
        # For CSV loader we need csv_path
        if not csv_path:
            with open(fallback_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            csv_path = cfg.get("data", {}).get("csv_path")
        if not csv_path:
            raise ValueError("Checkpoint indicates use_chembl=True but csv_path is missing.")
        # function returns (train, val, test, label_stats)
        _, _, test_loader, label_stats = create_data_loaders_from_chembl_csv(
            csv_path=str(csv_path),
            batch_size=batch_size,
            num_workers=num_workers,
            use_chembl=True,
        )
        return test_loader, label_stats

    else:
        # synthetic loader returns (train, val, test)
        _, _, test_loader = create_data_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            config_path=fallback_config_path,
        )
        return test_loader,None
    

def make_config_for_checkpoint(fallback_config_path: str, ckpt: dict) -> str:
    """
    Create a temporary config json that matches the checkpoint's target set.
    This prevents n_targets/targets mismatches when having multiple datasets/models.
    Returns path to temp config file.
    """
    with open(fallback_config_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    cfg = copy.deepcopy(base_cfg)

    ds = (ckpt.get("dataset_meta", {}) or {}) if isinstance(ckpt, dict) else {}
    target_names = ds.get("target_names", None)
    n_targets = ds.get("n_targets", None)

    if target_names is None or n_targets is None:
        # old checkpoint: best effort fallback to base config
        return fallback_config_path

    # Override targets to match checkpoint order exactly
    cfg["targets"] = {t: t for t in target_names}
    cfg.setdefault("model", {})
    cfg["model"]["n_targets"] = int(n_targets)

    # Also ensure data.use_chembl & csv_path match the checkpoint
    cfg.setdefault("data", {})
    cfg["data"]["use_chembl"] = bool(ds.get("use_chembl", cfg["data"].get("use_chembl", False)))
    if cfg["data"]["use_chembl"]:
        cfg["data"]["csv_path"] = ds.get("csv_path", cfg["data"].get("csv_path"))

    # Write temp file
    fd, tmp_path = tempfile.mkstemp(prefix="eval_cfg_", suffix=".json")
    Path(tmp_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return tmp_path


def pearson_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = torch.sqrt((yt**2).sum()) * torch.sqrt((yp**2).sum()) + 1e-8
    return float((yt * yp).sum() / denom)

def lifelines_ci(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = y_true.detach().cpu().numpy().astype(np.float64)
    yp = y_pred.detach().cpu().numpy().astype(np.float64)
    return float(concordance_index(yt, yp))

def flatten_measured(pred, target, mask):
    m = mask.bool()
    return pred[m].detach().cpu(), target[m].detach().cpu()

@torch.no_grad()
def evaluate_model_like_training(model, test_loader,label_stats=None):
    """
    Mirrors your train.py:
      preds, attn, _ = model(batch["molecules"])   -> preds [B,T]
      targets = batch["individual_affinities"]
      mask = batch.get("affinity_mask", ones)
    Computes:
      - measured-only MSE
      - measured-only CI (lifelines)
      - measured-only Pearson
      - per-target CI/Pearson too
    """
    model.eval()
    all_pred = []
    all_tgt = []
    all_mask = []
    all_loss_per_sample = []

    for batch in tqdm(test_loader, leave=False):
        molecules = batch["molecules"] 
        preds, _, _ = model(molecules)  # [B,T]

        targets = batch["individual_affinities"].to(DEVICE)  # [B,T]
        mask = batch.get("affinity_mask", None)

        if mask is None:
            # synthetic or loaders without a mask
            mask = torch.ones_like(targets, device=DEVICE)
        else:
            mask = mask.to(DEVICE)

        diff2 = (preds - targets) ** 2
        diff2_m = diff2 * mask

        # per-sample measured MSE = sum(diff2_m)/sum(mask) per row
        row_cnt = mask.sum(dim=1).clamp_min(1.0)
        row_mse = diff2_m.sum(dim=1) / row_cnt
        all_loss_per_sample.append(row_mse.detach().cpu())

        all_pred.append(preds.detach())
        all_tgt.append(targets.detach())
        all_mask.append(mask.detach())

    preds = torch.cat(all_pred, dim=0)
    targets = torch.cat(all_tgt, dim=0)
    mask = torch.cat(all_mask, dim=0)
    losses = torch.cat(all_loss_per_sample, dim=0)
    mask_bool = mask.bool()
    t_meas = targets[mask_bool]
    p_meas = preds[mask_bool]

    print("SCALE CHECK:",
        "targets(measured) mean/std/min/max =",
        float(t_meas.mean().item()), float(t_meas.std().item()),
        float(t_meas.min().item()), float(t_meas.max().item()))
    print("SCALE CHECK:",
        "preds(measured) mean/std/min/max =",
        float(p_meas.mean().item()), float(p_meas.std().item()),
        float(p_meas.min().item()), float(p_meas.max().item()))


    # boolean measured mask
    mask_bool = mask.bool()

    # global measured RMSE over ALL measured entries
    diff2_all = (preds - targets) ** 2
    mse_global = float(diff2_all[mask_bool].mean().item())
    rmse_global = float(torch.sqrt(torch.tensor(mse_global)).item())
    rmse_denorm = None
    if label_stats is not None:
        # target order must match config/targets
        targets_list = label_stats["targets"]  # same sorted unique_targets in loader
        means = torch.tensor([label_stats["target_stats"][str(t)]["mean"] for t in targets_list], dtype=torch.float32)
        stds  = torch.tensor([label_stats["target_stats"][str(t)]["std"]  for t in targets_list], dtype=torch.float32)

        # broadcast to [N,T]
        means = means.view(1, -1)
        stds  = stds.view(1, -1)

        preds_dn = preds.cpu() * stds + means
        targets_dn = targets.cpu() * stds + means
        diff2_dn = (preds_dn - targets_dn) ** 2
        mse_dn = float(diff2_dn[mask_bool.cpu()].mean().item())
        rmse_denorm = float(torch.sqrt(torch.tensor(mse_dn)).item())


    mse_rowmean = float(losses.mean().item())
    rmse_rowmean = float(torch.sqrt(torch.tensor(mse_rowmean)).item())

    # flatten for rank/corr
    flat_p = preds[mask_bool].detach().cpu()
    flat_t = targets[mask_bool].detach().cpu()

    pearson = pearson_corr(flat_t, flat_p)
    ci = lifelines_ci(flat_t, flat_p)


    # per-target
    T = targets.shape[1]
    per_target = []
    for i in range(T):
        p_i, t_i = flatten_measured(preds[:, i], targets[:, i], mask[:, i])
        if t_i.numel() < 2:
            per_target.append({"pearson": 0.0, "ci": 0.5})
        else:
            diff2_i = (p_i - t_i) ** 2
            rmse_i = float(torch.sqrt(diff2_i.mean()).item())

            per_target.append({
                "rmse": rmse_i,
                "pearson": pearson_corr(t_i, p_i),
                "ci": lifelines_ci(t_i, p_i),
            })

    metrics = {
        "rmse_measured_global": rmse_global,
        "rmse_measured_rowmean": rmse_rowmean,
        "pearson_measured": pearson,
        "ci_measured": ci,
        "per_target": per_target,
        "n_samples": int(targets.shape[0]),
        "n_targets": int(T),
        "measured_count_total": int(mask.sum().item()),
    }
    metrics["rmse_denormalized"] = rmse_denorm
    return metrics, preds.cpu(), targets.cpu(), mask.cpu(), losses

def save_plots(model_name: str, meta: dict, preds, targets, mask, losses, target_names):
    out_dir = PLOTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # measured flatten for scatter
    m = mask.bool()
    true_flat = targets[m].numpy()
    pred_flat = preds[m].numpy()

    plt.figure()
    plt.scatter(true_flat, pred_flat, alpha=0.35)
    plt.xlabel("True (measured only)")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} | {meta['variant']} {meta['backbone']}")
    plt.savefig(out_dir / "pred_vs_true_measured.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    rmse_per_sample = torch.sqrt(losses).numpy()
    plt.hist(rmse_per_sample, bins=60)
    plt.title(f"{model_name}: Measured-only per-sample RMSE distribution")
    plt.xlabel("RMSE")
    plt.ylabel("Count")
    plt.savefig(out_dir / "loss_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # per-target CI
    T = targets.shape[1]
    per_target_ci = []
    for i in range(T):
        mi = mask[:, i].bool()
        if mi.sum().item() < 2:
            per_target_ci.append(0.5)
        else:
            per_target_ci.append(lifelines_ci(targets[mi, i], preds[mi, i]))

    plt.figure(figsize=(8, 4))
    plt.bar(target_names, per_target_ci)
    plt.xlabel("Protein target")
    plt.ylabel("CI")
    plt.title(f"{model_name}: Per-target CI")
    plt.xticks(rotation=30, ha="right")
    plt.savefig(out_dir / "per_target_ci.png", dpi=150, bbox_inches="tight")
    plt.close()

def build_model_from_checkpoint(ckpt_obj, meta: dict, config_path: str):
    """
    Supports:
      - NEW format: dict with "model_state_dict"
      - OLD format: plain state_dict (OrderedDict)
    Instantiation matches train.py:
      ModelClass(config_path=..., model_type=<variant>)
    """
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state = ckpt_obj["model_state_dict"]
        variant = ckpt_obj.get("model_type_variant", meta["variant"])
        backbone = ckpt_obj.get("backbone", meta["backbone"])
    else:
        # old checkpoints saved as model.state_dict() only
        state = ckpt_obj
        variant = meta["variant"]
        backbone = meta["backbone"]

    module = _load_model_module(backbone)
    ModelClass = module.MultiTargetPredictor

    model = ModelClass(config_path=config_path, model_type=variant).to(DEVICE)
    model.load_state_dict(state, strict=True)
    return model, variant, backbone

def run_all_model_tests(config_path: str, num_workers: int = 0) -> dict:
    model_files = sorted(MODELS_DIR.glob("*.pt"))
    if not model_files:
        return {"ok": False, "message": f"No model checkpoints found in {MODELS_DIR}."}

    all_results = []
    failed = []

    run_meta = {
        "run_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(DEVICE),
        "models_dir": str(MODELS_DIR),
        "results_dir": str(RESULTS_DIR),
        "plots_dir": str(PLOTS_DIR),
        "fallback_config_path": str(config_path),
    }

    for model_path in model_files:
        meta = parse_model_name(model_path.name)
        model_name = model_path.stem

        if meta["variant"] == "unknown":
            failed.append({"model": model_name, "error": "Filename pattern not recognized"})
            continue

        row = {
            "model_name": model_name,
            "checkpoint_path": str(model_path),
            "parsed_name_meta": meta,
            "loaded_variant": None,
            "loaded_backbone": None,
            "dataset_meta": None,
            "metrics": None,
            "checkpoint_model_config": None,
            "checkpoint_training_config": None,
            "effective_config_path": None,
            "status": "pending",
            "error": None,
        }

        try:
            ckpt_obj = torch.load(model_path, map_location=DEVICE)

            # build a temp/effective config that matches checkpoint target set (if possible)
            effective_config_path = make_model_init_config_for_checkpoint(
                fallback_config_path=config_path,
                ckpt=ckpt_obj if isinstance(ckpt_obj, dict) else {}
            )

            row["effective_config_path"] = str(effective_config_path)
            debug_effective_config(effective_config_path, ckpt_obj if isinstance(ckpt_obj, dict) else {}, meta)

            # build correct test loader based on dataset origin (chembl vs synthetic)
            test_loader,label_stats = make_test_loader_for_checkpoint(
                ckpt=ckpt_obj if isinstance(ckpt_obj, dict) else {},
                fallback_config_path=effective_config_path,
                num_workers=num_workers,
                batch_size=meta["bs"],
            )

            # build model and evaluate
            model, variant, backbone = build_model_from_checkpoint(ckpt_obj, meta, effective_config_path)
            metrics, preds, targets, mask, losses = evaluate_model_like_training(model, test_loader)

            # gather extra checkpoint info if available
            if isinstance(ckpt_obj, dict):
                row["dataset_meta"] = ckpt_obj.get("dataset_meta", None)
                row["checkpoint_model_config"] = ckpt_obj.get("model_config", None)
                row["checkpoint_training_config"] = ckpt_obj.get("training_config", None)

            row["loaded_variant"] = variant
            row["loaded_backbone"] = backbone
            row["metrics"] = metrics
            row["status"] = "ok"

            # still keep per-model metrics.json (optional; you can remove if you want)
            model_result_dir = RESULTS_DIR / model_name
            model_result_dir.mkdir(parents=True, exist_ok=True)
            with open(model_result_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2, default=json_safe)

            # plots
            target_names = load_target_names(effective_config_path)
            save_plots(model_name, meta, preds, targets, mask, losses, target_names)

            all_results.append(row)

        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)
            failed.append({"model": model_name, "error": str(e)})
            all_results.append(row)

    # write one consolidated file for all models
    consolidated = {
        "run_meta": run_meta,
        "results": all_results,
        "failed": failed,
    }

    consolidated_json_path, consolidated_jsonl_path = write_all_results(
        all_results=consolidated,  # store as one object containing meta + results
        out_dir=RESULTS_DIR,
        basename="all_models_results"
    )

    # Keep the old "summary_path" semantics:
    # - previously it pointed to RESULTS_DIR / "test_summary.json"
    summary_path = str(RESULTS_DIR / "test_summary.json")

    old_style_summary = []
    for r in all_results:
        if r.get("status") != "ok":
            continue
        meta = r.get("parsed_name_meta", {})
        m = r.get("metrics", {}) or {}
        old_style_summary.append({
            "model_name": r.get("model_name"),
            **meta,
            "rmse_measured": m.get("rmse_measured_global"),
            "rmse_measured_rowmean": m.get("rmse_measured_rowmean"),
            "ci_measured": m.get("ci_measured"),
            "pearson_measured": m.get("pearson_measured"),
            "measured_count_total": m.get("measured_count_total"),
        })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(old_style_summary, f, indent=2, default=json_safe)

    return {
        "ok": True,

        # ✅ keys the UI expects
        "tested": sum(r.get("status") == "ok" for r in all_results),
        "failed": failed,
        "summary_path": summary_path,
        "results_dir": str(RESULTS_DIR),
        "plots_dir": str(PLOTS_DIR),

        # ✅ new keys (for table building)
        "tested_total": len(all_results),
        "tested_ok": sum(r.get("status") == "ok" for r in all_results),
        "failed_count": len(failed),
        "consolidated_json": consolidated_json_path,
        "consolidated_jsonl": consolidated_jsonl_path,
    }


