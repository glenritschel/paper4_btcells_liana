#!/usr/bin/env python3

import argparse
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--celltype-col", default="cell_type")
    ap.add_argument("--b-label", default="B_cell")
    ap.add_argument("--score-cols", nargs="+", required=True)
    ap.add_argument("--compute-umap", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ad = sc.read_h5ad(args.h5ad)

    # Subset B cells
    b = ad[ad.obs[args.celltype_col] == args.b_label].copy()

    # Create combined activation score
    b.obs["B_activation_combined"] = b.obs[args.score_cols].mean(axis=1)

    # Compute UMAP if needed
    if args.compute_umap or "X_umap" not in b.obsm:
        sc.pp.pca(b)
        sc.pp.neighbors(b)
        sc.tl.umap(b)

    # Plot
    sc.pl.umap(
        b,
        color="B_activation_combined",
        cmap="viridis",
        frameon=False,
        show=False
    )
    plt.savefig(args.out, dpi=300)
    plt.close()

    print("Wrote:", args.out)
    print("Activation stats:")
    x = b.obs["B_activation_combined"].to_numpy()
    print("  mean =", np.mean(x))
    print("  min  =", np.min(x))
    print("  max  =", np.max(x))

if __name__ == "__main__":
    main()

