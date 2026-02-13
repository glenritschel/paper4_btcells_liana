#!/usr/bin/env python3
"""
Partial correlation for LIANA rank-shift vs expression amplification.

Goal:
  Assess association between delta_rank and pair_amp_log2fc while controlling for:
    (A) lig_log2fc
    (B) rec_log2fc
    (C) both lig_log2fc and rec_log2fc

Method:
  - Residualize x and y via OLS on covariates
  - Compute Pearson and Spearman correlations on residuals
  - P-values from scipy (Pearson) and spearmanr (Spearman)

Usage:
  python scripts/partial_correlation.py \
    --input results/amplification_q1q4_210395/lr_amplification_gse210395_Blow_vs_gse210395_Bhigh.tsv \
    --x delta_rank \
    --y pair_amp_log2fc \
    --controls lig_log2fc rec_log2fc

Notes:
  - Spearman partial correlation is computed by rank-transforming variables first,
    then doing the same residualization + Pearson on residuals.
"""

import argparse
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def residualize(y: np.ndarray, X: Optional[np.ndarray]) -> np.ndarray:
    """
    Return residuals of y after OLS regression on X (with intercept).
    If X is None or has zero columns, residuals are y - mean(y).
    """
    y = np.asarray(y).astype(float).ravel()
    if X is None:
        return y - np.nanmean(y)

    X = _as_2d(np.asarray(X).astype(float))
    if X.shape[1] == 0:
        return y - np.nanmean(y)

    # Add intercept
    X1 = np.column_stack([np.ones(X.shape[0]), X])

    # OLS via least squares (handles non-square)
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ beta
    return y - y_hat


def pearson_with_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def spearman_with_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def partial_corr_pearson(x: np.ndarray, y: np.ndarray, Z: Optional[np.ndarray]) -> Tuple[float, float]:
    rx = residualize(x, Z)
    ry = residualize(y, Z)
    return pearson_with_p(rx, ry)


def partial_corr_spearman(x: np.ndarray, y: np.ndarray, Z: Optional[np.ndarray]) -> Tuple[float, float]:
    # Rank-transform x, y, and each covariate column in Z, then Pearson on residuals
    x_r = stats.rankdata(x, method="average")
    y_r = stats.rankdata(y, method="average")

    if Z is None:
        Z_r = None
    else:
        Z = _as_2d(Z)
        Z_r = np.column_stack([stats.rankdata(Z[:, j], method="average") for j in range(Z.shape[1])])

    rx = residualize(x_r, Z_r)
    ry = residualize(y_r, Z_r)
    return pearson_with_p(rx, ry)


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df


def select_complete_cases(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    before = len(df)
    out = df.dropna(subset=cols).copy()
    after = len(out)
    dropped = before - after
    if dropped > 0:
        print(f"[INFO] Dropped {dropped} rows with NA in {cols}", file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input TSV from lr_amplification_*.tsv")
    ap.add_argument("--x", default="delta_rank", help="X variable (default: delta_rank)")
    ap.add_argument("--y", default="pair_amp_log2fc", help="Y variable (default: pair_amp_log2fc)")
    ap.add_argument(
        "--controls",
        nargs="*",
        default=[],
        help="Covariate column(s). Examples: lig_log2fc rec_log2fc",
    )
    args = ap.parse_args()

    df = load_df(args.input)

    needed = [args.x, args.y] + list(args.controls)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] Missing required columns: {missing}\nAvailable: {df.columns.tolist()}")

    df0 = select_complete_cases(df, needed)

    x = df0[args.x].astype(float).to_numpy()
    y = df0[args.y].astype(float).to_numpy()

    Z = None
    if args.controls:
        Z = df0[list(args.controls)].astype(float).to_numpy()

    # Zero-order correlations
    sp0_rho, sp0_p = spearman_with_p(x, y)
    pe0_r, pe0_p = pearson_with_p(x, y)

    # Partial correlations
    psp_rho, psp_p = partial_corr_spearman(x, y, Z)
    ppe_r, ppe_p = partial_corr_pearson(x, y, Z)

    n = len(df0)
    ctrl_str = ", ".join(args.controls) if args.controls else "(none)"
    print(f"[INPUT] {args.input}")
    print(f"[VARS] n={n} x={args.x} y={args.y} controls={ctrl_str}\n")

    print("[ZERO-ORDER]")
    print(f"  Spearman rho = {sp0_rho:.6f}   p = {sp0_p:.3e}")
    print(f"  Pearson  r   = {pe0_r:.6f}   p = {pe0_p:.3e}\n")

    print("[PARTIAL (residualize on controls)]")
    print(f"  Partial Spearman (rank+resid) = {psp_rho:.6f}   p = {psp_p:.3e}")
    print(f"  Partial Pearson  (resid)      = {ppe_r:.6f}   p = {ppe_p:.3e}")


if __name__ == "__main__":
    main()

