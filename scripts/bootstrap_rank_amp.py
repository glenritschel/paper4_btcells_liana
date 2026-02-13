#!/usr/bin/env python3
"""
Bootstrap + permutation for association between delta_rank and pair_amp_log2fc (or any x/y).
Outputs:
  - bootstrap distributions (TSV)
  - empirical CI
  - permutation p-value
  - JSON summary
  - optional scatter plot

Example:
  python scripts/bootstrap_rank_amp.py \
    --input results/amplification_gse210395_blow_vs_bhigh/lr_amplification_gse210395_blow_vs_bhigh.tsv \
    --x delta_rank \
    --y pair_amp_log2fc \
    --method spearman \
    --n-bootstrap 20000 \
    --n-perm 20000 \
    --seed 0 \
    --outdir results/robustness_gse210395 \
    --prefix gse210395_rank_amp \
    --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def corr(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float]:
    if method == "spearman":
        r, p = stats.spearmanr(x, y, nan_policy="omit")
        return float(r), float(p)
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    raise ValueError(f"Unknown method: {method}")


def bootstrap_corr(x: np.ndarray, y: np.ndarray, method: str, n: int, rng: np.random.Generator) -> np.ndarray:
    m = len(x)
    out = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        out[i], _ = corr(x[idx], y[idx], method)
    return out


def perm_corr(x: np.ndarray, y: np.ndarray, method: str, n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n, dtype=float)
    for i in range(n):
        yp = rng.permutation(y)
        out[i], _ = corr(x, yp, method)
    return out


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = np.nanpercentile(samples, 100 * (alpha / 2))
    hi = np.nanpercentile(samples, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="robustness")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")
    if args.x not in df.columns or args.y not in df.columns:
        raise SystemExit(f"Missing columns. Need x={args.x}, y={args.y}")

    work = df[[args.x, args.y]].copy()
    work[args.x] = pd.to_numeric(work[args.x], errors="coerce")
    work[args.y] = pd.to_numeric(work[args.y], errors="coerce")
    work = work.dropna(axis=0, how="any")

    x = work[args.x].to_numpy()
    y = work[args.y].to_numpy()

    if len(x) < 5:
        raise SystemExit(f"Too few rows after dropna: {len(x)}")

    rng = np.random.default_rng(args.seed)

    r_obs, p_obs = corr(x, y, args.method)

    boot = bootstrap_corr(x, y, args.method, args.n_bootstrap, rng)
    ci_lo, ci_hi = percentile_ci(boot, alpha=args.alpha)

    perm = perm_corr(x, y, args.method, args.n_perm, rng)
    # Two-sided empirical p-value
    p_perm = float((np.sum(np.abs(perm) >= np.abs(r_obs)) + 1) / (len(perm) + 1))

    prefix = args.prefix
    boot_path = outdir / f"{prefix}.bootstrap.tsv"
    perm_path = outdir / f"{prefix}.perm.tsv"
    json_path = outdir / f"{prefix}.summary.json"
    plot_path = outdir / f"{prefix}.scatter.png"

    pd.DataFrame({"r": boot}).to_csv(boot_path, sep="\t", index=False)
    pd.DataFrame({"r": perm}).to_csv(perm_path, sep="\t", index=False)

    summary: Dict[str, Any] = {
        "input": args.input,
        "x": args.x,
        "y": args.y,
        "method": args.method,
        "n_rows": int(len(x)),
        "r_observed": r_obs,
        "p_observed_parametric": p_obs,
        "bootstrap": {
            "n": args.n_bootstrap,
            "alpha": args.alpha,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "mean": float(np.nanmean(boot)),
            "median": float(np.nanmedian(boot)),
            "std": float(np.nanstd(boot, ddof=1)),
        },
        "permutation": {
            "n": args.n_perm,
            "p_empirical_two_sided": p_perm,
            "null_mean": float(np.nanmean(perm)),
            "null_std": float(np.nanstd(perm, ddof=1)),
        },
        "seed": args.seed,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    if args.plot:
        plt.figure()
        plt.scatter(x, y, s=12)
        plt.xlabel(args.x)
        plt.ylabel(args.y)
        plt.title(f"{args.method}: r={r_obs:.3f}, perm_p={p_perm:.4g}, n={len(x)}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Wrote: {plot_path}")

    print(f"Wrote: {boot_path}")
    print(f"Wrote: {perm_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

