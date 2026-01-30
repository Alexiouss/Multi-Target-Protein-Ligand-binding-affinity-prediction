import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", default="results/analysis/pca")
    ap.add_argument("--n_components", type=int, default=2)
    ap.add_argument("--standardize", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.npz, allow_pickle=True)
    X = data["embeddings"]

    if args.standardize:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=args.n_components, random_state=0)
    Z = pca.fit_transform(X)

    # Save reduced
    out_npz = os.path.join(args.out_dir, "pca_reduced.npz")
    np.savez_compressed(out_npz, Z=Z, explained_variance_ratio=pca.explained_variance_ratio_)
    print("Saved:", out_npz)

    # Plot EVR
    plt.figure()
    evr = pca.explained_variance_ratio_
    plt.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), marker="o")
    plt.xlabel("Component")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pca_explained_variance.png"), dpi=200)
    plt.close()

    # Scatter if 2D
    if args.n_components == 2:
        mean_pred = data["mean_pred"] if "mean_pred" in data.files else None
        plt.figure()
        if mean_pred is None:
            plt.scatter(Z[:, 0], Z[:, 1], s=8)
            plt.title("PCA (2D) of embeddings")
        else:
            sc = plt.scatter(Z[:, 0], Z[:, 1], c=mean_pred, s=8)
            plt.colorbar(sc, label="Mean predicted pKd")
            plt.title("PCA (2D) coloured by mean predicted pKd")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "pca_scatter.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    main()
