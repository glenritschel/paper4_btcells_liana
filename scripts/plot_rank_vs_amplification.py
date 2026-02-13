#!/usr/bin/env python3
"""
Scatter: delta_rank vs pair_amp_log2fc

- Reads an amplification TSV (from your pipeline)
- Plots scatter + OLS fit line
- Annotates Spearman rho/p and Pearson r/p
- Writes PNG + PDF to --outdir

Example:
  python scripts/plot_rank_vs_amplification.py \
    --input results/amplification_q1q4_210395/lr_amplification_gse210395_Blow_vs_gse210395_Bhigh.tsv \
    --outdir results/figures \
    --label "GSE210395 Blow vs Bhigh"
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


def finite_xy(df: pd.DataFrame, xcol: str, ycol: str) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Simple OLS y = a + b*x via np.polyfit.
    Returns: xgrid, yhat, intercept, slope
    """
    if len(x) < 2:
        return np.array([]), np.array([]), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)  # NOTE: polyfit returns [m, b]
    xgrid = np.linspace(np.min(x), np.max(x), 200)
    yhat = slope * xgrid + intercept
    return xgrid, yhat, intercept, slope


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Amplification TSV with delta_rank and pair_amp_log2fc columns")
    ap.add_argument("--outdir", required=True, help="Output directory for figures")
    ap.add_argument("--xcol", default="delta_rank")
    ap.add_argument("--ycol", default="pair_amp_log2fc")
    ap.add_argument("--label", default="", help="Optional label shown in title")
    ap.add_argument("--basename", default="rank_vs_amplification", help="Output filename stem (no extension)")
    ap.add_argument("--alpha", type=float, default=0.7, help="Point alpha")
    ap.add_argument("--s", type=float, default=18, help="Point size")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")
    if args.xcol not in df.columns or args.ycol not in df.columns:
        raise SystemExit(
            f"Missing required columns. Need '{args.xcol}' and '{args.ycol}'. "
            f"Have: {list(df.columns)}"
        )

    x, y = finite_xy(df, args.xcol, args.ycol)
    n = len(x)
    if n < 3:
        raise SystemExit(f"Not enough finite points to plot/correlate (n={n}).")

    # Correlations
    rho, rho_p = spearmanr(x, y)
    r, r_p = pearsonr(x, y)

    # Fit line
    xgrid, yhat, intercept, slope = fit_line(x, y)

    # ---- Plot ----
    fig = plt.figure(figsize=(7.5, 5.5))
    ax = plt.gca()

    ax.scatter(x, y, alpha=args.alpha, s=args.s)

    # OLS fit line
    if len(xgrid) > 0:
        ax.plot(xgrid, yhat, linewidth=2)

    # Reference lines (0)
    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)

    title = "Rank shift vs expression amplification"
    if args.label:
        title += f" ({args.label})"
    ax.set_title(title)

    ax.set_xlabel(args.xcol)
    ax.set_ylabel(args.ycol)

    # Annotation block
    ann = (
        f"n={n}\n"
        f"Spearman ρ={rho:.3f}, p={rho_p:.3g}\n"
        f"Pearson r={r:.3f}, p={r_p:.3g}\n"
        f"OLS: y={intercept:.3f} + {slope:.3f}x"
    )
    ax.text(
        0.02,
        0.98,
        ann,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Tight layout and save
    plt.tight_layout()

    out_png = os.path.join(args.outdir, args.basename + ".png")
    out_pdf = os.path.join(args.outdir, args.basename + ".pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"[OK] Wrote: {out_png}")
    print(f"[OK] Wrote: {out_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    main()

