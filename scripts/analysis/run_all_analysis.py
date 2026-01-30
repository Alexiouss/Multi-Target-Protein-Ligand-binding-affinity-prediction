import os
import re
import sys
import json
import glob
import argparse
import subprocess
from datetime import datetime
import tempfile
import torch


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def guess_from_filename(ckpt_name: str):
    """
    Tries to infer backbone + model_type from checkpoint filename.
    Expected patterns somewhere in name: 'gcn' or 'gine', 'quantum' or 'classical'.

    If not found, returns None and the runner will skip (or you can add fallback).
    """
    name = ckpt_name.lower()

    backbone = None
    if "gcn" in name:
        backbone = "gcn"
    elif "gine" in name or "gin" in name:
        # careful: sometimes people write "gin" in name
        backbone = "gine"

    model_type = None
    if "quantum" in name:
        model_type = "quantum"
    elif "classical" in name:
        model_type = "classical"

    return backbone, model_type

def run(cmd, cwd=None):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def resolve_config_for_checkpoint(ckpt_path: str, project_root: str, fallback_config_path: str) -> str:
    """
    Return a config path to use for THIS checkpoint.

    Priority:
      1) ckpt["config_path"] if exists on disk
      2) build a temp config by loading fallback config and overwriting with ckpt["model_config"] (if present)
      3) fallback_config_path (as last resort)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1) if training saved the config path
    cfg_path = ckpt.get("config_path", None)
    if cfg_path:
        # handle relative paths
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(project_root, cfg_path)
        if os.path.exists(cfg_path):
            return cfg_path

    # 2) if training saved model_config, synthesize a temp config
    model_cfg = ckpt.get("model_config", None)
    if isinstance(model_cfg, dict) and os.path.exists(fallback_config_path):
        base = read_json(fallback_config_path)

        # many of your configs store these under base["model"]
        if "model" not in base or not isinstance(base["model"], dict):
            base["model"] = {}

        # overwrite only keys that matter for instantiation
        for k, v in model_cfg.items():
            base["model"][k] = v

        # write temp config next to runner output (or in system temp)
        fd, tmp_path = tempfile.mkstemp(prefix="cfg_from_ckpt_", suffix=".json")
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2, ensure_ascii=False)

        return tmp_path

    # 3) last resort
    return fallback_config_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".", help="Project root (contains src/)")
    ap.add_argument("--config", default="config_ui_run.json", help="UI config json (path)")
    ap.add_argument("--models_dir", default="results/models", help="Directory with checkpoints (*.pt)")
    ap.add_argument("--out_root", default="results/analysis", help="Where to save analysis outputs")
    ap.add_argument("--max_checkpoints", type=int, default=None, help="For quick runs")
    ap.add_argument("--max_batches", type=int, default=None, help="Limit extraction for speed")
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_pca_dim", type=int, default=50)
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--standardize", action="store_true", help="Standardize embeddings for PCA/tSNE/KMeans")
    ap.add_argument("--skip_tsne", action="store_true")
    ap.add_argument("--skip_clusters", action="store_true")
    ap.add_argument("--only_best", action="store_true", help="Only checkpoints containing 'best' in filename")
    args = ap.parse_args()

    project_root = os.path.abspath(args.project_root)
    config_path = args.config if os.path.isabs(args.config) else os.path.join(project_root, args.config)
    models_dir = args.models_dir if os.path.isabs(args.models_dir) else os.path.join(project_root, args.models_dir)
    out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(project_root, args.out_root)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models dir not found: {models_dir}")

    fallback_config_path = config_path  # your UI config becomes the fallback
    cfg = read_json(fallback_config_path)
    target_names = list(cfg.get("targets", {}).keys())

    print("Config:", config_path)
    print("Targets:", target_names if target_names else "(none found in config)")

    ensure_dir(out_root)

    # Collect checkpoints
    ckpts = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
    if args.only_best:
        ckpts = [c for c in ckpts if "best" in os.path.basename(c).lower()]

    if not ckpts:
        print("No checkpoints found.")
        return

    if args.max_checkpoints is not None:
        ckpts = ckpts[: args.max_checkpoints]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_tag = f"batch_{timestamp}"
    batch_out = os.path.join(out_root, batch_tag)
    ensure_dir(batch_out)

    # Store a run manifest for traceability
    manifest = {
        "timestamp": timestamp,
        "config_path": config_path,
        "models_dir": models_dir,
        "out_root": out_root,
        "n_checkpoints": len(ckpts),
        "settings": {
            "max_batches": args.max_batches,
            "tsne_perplexity": args.tsne_perplexity,
            "tsne_pca_dim": args.tsne_pca_dim,
            "clusters": args.clusters,
            "standardize": args.standardize,
            "skip_tsne": args.skip_tsne,
            "skip_clusters": args.skip_clusters,
        },
        "checkpoints": [],
    }

    # Script paths (relative to project root)
    extract_script = os.path.join(project_root, "scripts", "extract_embeddings_and_outputs.py")
    pca_script = os.path.join(project_root, "scripts", "analyze_pca_embeddings.py")
    tsne_script = os.path.join(project_root, "scripts", "analyze_tsne_embeddings.py")
    cluster_script = os.path.join(project_root, "scripts", "cluster_by_activity.py")

    for s in [extract_script, pca_script, tsne_script, cluster_script]:
        if not os.path.exists(s):
            raise FileNotFoundError(f"Missing script: {s}")

    # Run per checkpoint
    for ckpt in ckpts:
        ckpt_name = os.path.basename(ckpt)
        per_ckpt_config = resolve_config_for_checkpoint(
            ckpt_path=ckpt,
            project_root=project_root,
            fallback_config_path=fallback_config_path,
        )
        backbone, model_type = guess_from_filename(ckpt_name)

        if backbone is None or model_type is None:
            print(f"\n[SKIP] Could not infer backbone/model_type from filename: {ckpt_name}")
            print("       Ensure name contains: 'gcn' or 'gine' AND 'quantum' or 'classical'")
            continue

        # Out folder layout:
        # results/analysis/batch_xxx/{model_type}/{backbone}/{ckpt_stem}/...
        ckpt_stem = os.path.splitext(ckpt_name)[0]
        run_dir = os.path.join(batch_out, model_type, backbone, ckpt_stem)
        ensure_dir(run_dir)

        npz_path = os.path.join(run_dir, "test_outputs.npz")

        # 1) Extraction
        cmd = [
            sys.executable, extract_script,
            "--project_root", project_root,
            "--config", per_ckpt_config,
            "--checkpoint", ckpt,
            "--backbone", backbone,
            "--model_type", model_type,
            "--split", "test",
            "--out_npz", npz_path
        ]
        if args.max_batches is not None:
            cmd += ["--max_batches", str(args.max_batches)]

        run(cmd)

        # 2) PCA
        pca_out = os.path.join(run_dir, "pca")
        ensure_dir(pca_out)
        cmd = [
            sys.executable, pca_script,
            "--npz", npz_path,
            "--out_dir", pca_out,
            "--n_components", "2",
        ]
        if args.standardize:
            cmd.append("--standardize")
        run(cmd)

        # 3) t-SNE
        if not args.skip_tsne:
            tsne_out = os.path.join(run_dir, "tsne")
            ensure_dir(tsne_out)
            cmd = [
                sys.executable, tsne_script,
                "--npz", npz_path,
                "--out_dir", tsne_out,
                "--pca_dim", str(args.tsne_pca_dim),
                "--perplexity", str(args.tsne_perplexity),
                "--seed", "0",
            ]
            if args.standardize:
                cmd.append("--standardize")
            run(cmd)

        # 4) Clustering (two modes)
        if not args.skip_clusters:
            cl_out = os.path.join(run_dir, "clustering")
            ensure_dir(cl_out)

            # (a) Activity quantile bins
            cmd = [
                sys.executable, cluster_script,
                "--npz", npz_path,
                "--out_dir", os.path.join(cl_out, "activity_bins"),
                "--mode", "activity_bins",
                "--n_clusters", str(args.clusters),
            ]
            if args.standardize:
                cmd.append("--standardize")
            run(cmd)

            # (b) KMeans on embeddings
            cmd = [
                sys.executable, cluster_script,
                "--npz", npz_path,
                "--out_dir", os.path.join(cl_out, "kmeans_embeddings"),
                "--mode", "kmeans_embeddings",
                "--n_clusters", str(args.clusters),
            ]
            if args.standardize:
                cmd.append("--standardize")
            run(cmd)

        manifest["checkpoints"].append({
            "checkpoint": ckpt,
            "name": ckpt_name,
            "backbone": backbone,
            "model_type": model_type,
            "run_dir": run_dir,
            "npz": npz_path,
            "config_used": per_ckpt_config,
        })

    # Save manifest
    manifest_path = os.path.join(batch_out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\n✅ DONE.")
    print("Batch output:", batch_out)
    print("Manifest:", manifest_path)

if __name__ == "__main__":
    main()
