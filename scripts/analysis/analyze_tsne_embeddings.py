import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", default="results/analysis/tsne")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.npz, allow_pickle=True)
    X = data["embeddings"]

    if args.standardize:
        X = StandardScaler().fit_transform(X)

    # PCA pre-reduction (stabilizes & speeds t-SNE)
    if args.pca_dim is not None and args.pca_dim < X.shape[1]:
        Xr = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(X)
    else:
        Xr = X

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )
    Z = tsne.fit_transform(Xr)

    np.savez_compressed(os.path.join(args.out_dir, "tsne_2d.npz"), Z=Z)
    print("Saved:", os.path.join(args.out_dir, "tsne_2d.npz"))

    # Plot 1: colour by mean predicted pKd
    mean_pred = data["mean_pred"] if "mean_pred" in data.files else None
    plt.figure()
    if mean_pred is None:
        plt.scatter(Z[:, 0], Z[:, 1], s=8)
        plt.title("t-SNE (2D) of embeddings")
    else:
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=mean_pred, s=8)
        plt.colorbar(sc, label="Mean predicted pKd")
        plt.title("t-SNE (colour = mean predicted pKd)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "tsne_mean_pkd.png"), dpi=200)
    plt.close()

    # Plot 2: colour by attention-dominant target
    if "dom_target_idx" in data.files:
        dom = data["dom_target_idx"]
        plt.figure()
        plt.scatter(Z[:, 0], Z[:, 1], c=dom, s=8)
        plt.colorbar(label="Dominant attention index")
        plt.title("t-SNE (colour = attention-dominant target)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "tsne_dom_target.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    main()
