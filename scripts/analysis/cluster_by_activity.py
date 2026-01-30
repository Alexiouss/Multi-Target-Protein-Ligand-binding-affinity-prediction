import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def quantile_bins(x, n_bins):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    # avoid duplicates
    edges = np.unique(edges)
    labels = np.digitize(x, edges[1:-1], right=True)
    return labels, edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", default="results/analysis/clustering")
    ap.add_argument("--mode", choices=["activity_bins", "kmeans_embeddings"], default="activity_bins")
    ap.add_argument("--n_clusters", type=int, default=5)
    ap.add_argument("--standardize", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.npz, allow_pickle=True)

    X = data["embeddings"]
    mean_pred = data["mean_pred"] if "mean_pred" in data.files else data["predictions"].mean(axis=1)
    attn = data["attention"] if "attention" in data.files else None

    if args.standardize:
        Xs = StandardScaler().fit_transform(X)
    else:
        Xs = X

    if args.mode == "activity_bins":
        cluster_id, edges = quantile_bins(mean_pred, args.n_clusters)
        title = f"Activity quantile bins (K={args.n_clusters})"
    else:
        km = KMeans(n_clusters=args.n_clusters, n_init="auto", random_state=0)
        cluster_id = km.fit_predict(Xs)
        edges = None
        title = f"KMeans on embeddings (K={args.n_clusters})"

    # --- Summaries per cluster
    clusters = np.unique(cluster_id)
    mean_pkd_per_cluster = []
    size_per_cluster = []
    mean_attn_per_cluster = []

    for c in clusters:
        idx = np.where(cluster_id == c)[0]
        size_per_cluster.append(len(idx))
        mean_pkd_per_cluster.append(mean_pred[idx].mean())

        if attn is not None:
            mean_attn_per_cluster.append(attn[idx].mean(axis=0))

    mean_pkd_per_cluster = np.array(mean_pkd_per_cluster)

    # Plot cluster sizes
    plt.figure()
    plt.bar([str(c) for c in clusters], size_per_cluster)
    plt.title(title + " — cluster sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "cluster_sizes.png"), dpi=200)
    plt.close()

    # Plot mean activity per cluster
    plt.figure()
    plt.bar([str(c) for c in clusters], mean_pkd_per_cluster)
    plt.title(title + " — mean predicted activity")
    plt.xlabel("Cluster")
    plt.ylabel("Mean predicted pKd")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "cluster_mean_pkd.png"), dpi=200)
    plt.close()

    # If attention exists, heatmap: cluster x target mean attention
    if attn is not None and len(mean_attn_per_cluster) > 0:
        A = np.vstack(mean_attn_per_cluster)  # [K, T]
        plt.figure(figsize=(8, 4))
        plt.imshow(A, aspect="auto")
        plt.colorbar(label="Mean attention")
        plt.yticks(range(len(clusters)), [f"c={c}" for c in clusters])

        if "target_names" in data.files:
            names = list(data["target_names"])
            plt.xticks(range(len(names)), names, rotation=45, ha="right")
        else:
            plt.xticks(range(A.shape[1]), [f"t{i}" for i in range(A.shape[1])], rotation=45, ha="right")

        plt.title(title + " — mean attention per cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "cluster_attention_heatmap.png"), dpi=200)
        plt.close()

    # Save numeric summary
    summary = {
        "mode": args.mode,
        "n_clusters": int(args.n_clusters),
        "clusters": clusters.tolist(),
        "cluster_sizes": [int(x) for x in size_per_cluster],
        "cluster_mean_pred_pkd": mean_pkd_per_cluster.tolist(),
    }
    if edges is not None:
        summary["activity_bin_edges"] = edges.tolist()

    out_json = os.path.join(args.out_dir, "cluster_summary.json")
    import json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", out_json)

if __name__ == "__main__":
    main()
