#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr, pearsonr


def dense_col(X) -> np.ndarray:
    # Handles sparse and dense
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .h5ad")
    ap.add_argument("--cell-type-col", default="cell_type")
    ap.add_argument("--cell-type", default="B_cell")
    ap.add_argument("--score-col", required=True, help="obs column (e.g., b_activation)")
    ap.add_argument("--gene-prefix", default="HLA-D", help="gene prefix to average (default HLA-D)")
    ap.add_argument("--layer", default=None, help="Optional layer to use instead of .X")
    args = ap.parse_args()

    ad = sc.read_h5ad(args.input)
    if args.cell_type_col not in ad.obs:
        raise RuntimeError(f"Missing obs['{args.cell_type_col}']")
    if args.score_col not in ad.obs:
        raise RuntimeError(f"Missing obs['{args.score_col}']")

    b = ad[ad.obs[args.cell_type_col] == args.cell_type].copy()
    genes = [g for g in b.var_names if str(g).startswith(args.gene_prefix)]
    if not genes:
        raise RuntimeError(f"No genes with prefix '{args.gene_prefix}' found.")

    X = b.layers[args.layer] if args.layer else b.X
    Xg = dense_col(b[:, genes].layers[args.layer] if args.layer else b[:, genes].X)
    mhc_mean = Xg.mean(axis=1)

    score = b.obs[args.score_col].astype(float).to_numpy()

    sp_r, sp_p = spearmanr(score, mhc_mean, nan_policy="omit")
    pe_r, pe_p = pearsonr(score[~np.isnan(score)], mhc_mean[~np.isnan(score)])

    print(f"[INPUT] {args.input}")
    print(f"[CELLTYPE] {args.cell_type} n={b.n_obs}")
    print(f"[GENES] prefix={args.gene_prefix} n_genes={len(genes)}")
    print(f"[CORR] Spearman rho={sp_r:.6f} p={sp_p:.3e}")
    print(f"[CORR] Pearson  r ={pe_r:.6f} p={pe_p:.3e}")


if __name__ == "__main__":
    main()

