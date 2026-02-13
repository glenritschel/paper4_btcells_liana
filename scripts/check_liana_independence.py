#!/usr/bin/env python3
"""
Check whether rank shifts track LIANA score deltas directly.

Compute:
  - delta_score = score_b - score_a
  - corr(delta_rank, delta_score)
  - corr(pair_amp_log2fc, delta_score)
  - partial corr(delta_rank, pair_amp_log2fc | delta_score)  (residual approach)

You must provide:
  --overlap (from 07_compare_liana_datasets.py)
  --amp (from amplification step)
  --join-cols (shared keys, usually ligand/receptor identifiers)

Example:
  python scripts/check_liana_independence.py \
    --overlap results/compare_gse210395_blow_vs_bhigh/gse210395_blow_vs_bhigh.overlap.tsv \
    --amp results/amplification_gse210395_blow_vs_bhigh/lr_amplification_gse210395_blow_vs_bhigh.tsv \
    --score-a colA_score \
    --score-b colB_score \
    --join-cols ligand receptor \
    --outdir results/independence_gse210395 \
    --prefix gse210395
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(y, X, missing="drop").fit()
    return fit.resid


def spearman(x, y) -> Dict[str, float]:
    r, p = stats.spearmanr(x, y, nan_policy="omit")
    return {"r": float(r), "p": float(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", required=True)
    ap.add_argument("--amp", required=True)
    ap.add_argument("--score-a", required=True)
    ap.add_argument("--score-b", required=True)
    ap.add_argument("--join-cols", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="independence")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ov = pd.read_csv(args.overlap, sep="\t")
    amp = pd.read_csv(args.amp, sep="\t")

    for c in args.join_cols + ["delta_rank", "pair_amp_log2fc", args.score_a, args.score_b]:
        if c not in ov.columns and c not in amp.columns:
            # delta_rank likely in overlap; pair_amp_log2fc likely in amp
            pass

    # Build merged table
    need_ov = args.join_cols + ["delta_rank", args.score_a, args.score_b]
    need_amp = args.join_cols + ["pair_amp_log2fc"]

    for c in need_ov:
        if c not in ov.columns:
            raise SystemExit(f"Missing in overlap: {c}")
    for c in need_amp:
        if c not in amp.columns:
            raise SystemExit(f"Missing in amp: {c}")

    m = ov[need_ov].merge(amp[need_amp], on=args.join_cols, how="inner")

    # numeric
    for c in ["delta_rank", "pair_amp_log2fc", args.score_a, args.score_b]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(axis=0, how="any")

    m["delta_score"] = m[args.score_b] - m[args.score_a]

    x = m["delta_rank"].to_numpy()
    y = m["pair_amp_log2fc"].to_numpy()
    s = m["delta_score"].to_numpy()

    # Basic correlations
    out: Dict[str, Any] = {
        "n_rows": int(len(m)),
        "corr_delta_rank_vs_amp": spearman(x, y),
        "corr_delta_rank_vs_delta_score": spearman(x, s),
        "corr_amp_vs_delta_score": spearman(y, s),
    }

    # Partial: delta_rank vs amp controlling delta_score
    rx = resid(x, s.reshape(-1, 1))
    ry = resid(y, s.reshape(-1, 1))
    out["partial_corr_delta_rank_vs_amp_given_delta_score"] = spearman(rx, ry)

    # Save
    prefix = args.prefix
    merged_path = outdir / f"{prefix}.merged.tsv"
    json_path = outdir / f"{prefix}.summary.json"
    m.to_csv(merged_path, sep="\t", index=False)
    json_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"Wrote: {merged_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

