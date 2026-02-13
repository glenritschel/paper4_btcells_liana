#!/usr/bin/env python3
"""
spearman_correlation.py

Compute correlation between delta_rank and pair_amp_log2fc:
- prints a compact summary line (good for tables/logs)
- prints Spearman correlation matrix (optional)
- outputs JSON with metrics (optional)
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def finite_pair_df(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    keep = np.isfinite(x) & np.isfinite(y)
    out = df.loc[keep, [xcol, ycol]].copy()
    out[xcol] = pd.to_numeric(out[xcol], errors="coerce")
    out[ycol] = pd.to_numeric(out[ycol], errors="coerce")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute Spearman/Pearson correlation between delta_rank and pair_amp_log2fc."
    )
    ap.add_argument("--input", required=True, help="Path to lr_amplification TSV file")
    ap.add_argument(
        "--xcol", default="delta_rank", help="X column (default: delta_rank)"
    )
    ap.add_argument(
        "--ycol",
        default="pair_amp_log2fc",
        help="Y column (default: pair_amp_log2fc)",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write JSON summary (e.g., results/.../spearman.json)",
    )
    ap.add_argument(
        "--no-matrix",
        action="store_true",
        help="Do not print the correlation matrix; only print summary line.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t")

    for c in (args.xcol, args.ycol):
        if c not in df.columns:
            raise ValueError(
                f"Missing required column '{c}'. Available columns: {df.columns.tolist()}"
            )

    pair = finite_pair_df(df, args.xcol, args.ycol)
    n = int(len(pair))

    if n < 3:
        raise ValueError(f"Not enough finite rows for correlation (n={n}).")

    x = pair[args.xcol].to_numpy()
    y = pair[args.ycol].to_numpy()

    # Spearman
    sp = spearmanr(x, y, nan_policy="omit")
    spearman_rho = float(sp.statistic)
    spearman_p = float(sp.pvalue)

    # Pearson (often useful as a secondary sanity check)
    pr = pearsonr(x, y)
    pearson_r = float(pr.statistic)
    pearson_p = float(pr.pvalue)

    # --- 1) Summary line (single line, easy to grep / paste into notes) ---
    # Convention: show rho and p in scientific notation.
    print(
        f"[SUMMARY] n={n} x={args.xcol} y={args.ycol} "
        f"spearman_rho={spearman_rho:.6f} spearman_p={spearman_p:.3e} "
        f"pearson_r={pearson_r:.6f} pearson_p={pearson_p:.3e}"
    )

    # --- 2) Correlation matrix (what you were printing before) ---
    if not args.no_matrix:
        print("\nSpearman correlation matrix:")
        print(pair[[args.xcol, args.ycol]].corr(method="spearman"))

    # --- 3) JSON output (machine-readable for reports / Makefile targets) ---
    if args.out_json:
        payload = {
            "inputs": {
                "input": args.input,
                "xcol": args.xcol,
                "ycol": args.ycol,
            },
            "n": n,
            "spearman": {"rho": spearman_rho, "p": spearman_p},
            "pearson": {"r": pearson_r, "p": pearson_p},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] Wrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()

