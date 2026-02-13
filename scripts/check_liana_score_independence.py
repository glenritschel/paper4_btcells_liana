#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def spearman(x, y):
    r, p = stats.spearmanr(x, y, nan_policy="omit")
    return {"r": float(r), "p": float(p)}

def resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(y, X, missing="drop").fit()
    return fit.resid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", required=True, help="compare_*overlap.tsv containing lr_id + delta_rank")
    ap.add_argument("--amp", required=True, help="lr_amplification_*.tsv containing lr_id + pair_amp_log2fc")
    ap.add_argument("--liana-a", required=True, help="LIANA top_bt for condition A (e.g., Blow)")
    ap.add_argument("--liana-b", required=True, help="LIANA top_bt for condition B (e.g., Bhigh)")
    ap.add_argument("--score-col", default="lrscore", help="Score column to use from LIANA")
    ap.add_argument("--join-cols", nargs="+", default=["lr_id"], help="Join keys, default lr_id")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="independence")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ov = pd.read_csv(args.overlap, sep="\t")
    amp = pd.read_csv(args.amp, sep="\t")
    a = pd.read_csv(args.liana_a, sep="\t")
    b = pd.read_csv(args.liana_b, sep="\t")

    # Require lr_id in LIANA; if not present, build it from components
    def ensure_lr_id(df):
        if "lr_id" in df.columns:
            return df
        req = {"source","target","ligand_complex","receptor_complex"}
        if not req.issubset(df.columns):
            raise SystemExit("LIANA file lacks lr_id and cannot build it (missing source/target/ligand_complex/receptor_complex).")
        df = df.copy()
        df["lr_id"] = df["source"].astype(str) + "|" + df["target"].astype(str) + "|" + df["ligand_complex"].astype(str) + "|" + df["receptor_complex"].astype(str)
        return df

    a = ensure_lr_id(a)
    b = ensure_lr_id(b)

    for df, name in [(a,"liana-a"), (b,"liana-b")]:
        if args.score_col not in df.columns:
            raise SystemExit(f"{name} missing score column: {args.score_col}")

    # select minimal
    a2 = a[args.join_cols + [args.score_col]].copy().rename(columns={args.score_col: "score_a"})
    b2 = b[args.join_cols + [args.score_col]].copy().rename(columns={args.score_col: "score_b"})

    # merge everything
    m = ov.merge(amp, on=args.join_cols, how="inner") \
          .merge(a2, on=args.join_cols, how="inner") \
          .merge(b2, on=args.join_cols, how="inner")

    for c in ["delta_rank", "pair_amp_log2fc", "score_a", "score_b"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna()

    m["delta_score"] = m["score_b"] - m["score_a"]

    x = m["delta_rank"].to_numpy()
    y = m["pair_amp_log2fc"].to_numpy()
    s = m["delta_score"].to_numpy()

    out = {
        "n_rows": int(len(m)),
        "score_col": args.score_col,
        "corr_delta_rank_vs_pair_amp": spearman(x, y),
        "corr_delta_rank_vs_delta_score": spearman(x, s),
        "corr_pair_amp_vs_delta_score": spearman(y, s),
        "partial_corr_delta_rank_vs_pair_amp_given_delta_score": spearman(resid(x, s.reshape(-1,1)), resid(y, s.reshape(-1,1))),
    }

    merged_path = outdir / f"{args.prefix}.merged.tsv"
    json_path = outdir / f"{args.prefix}.summary.json"
    m.to_csv(merged_path, sep="\t", index=False)
    json_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"Wrote: {merged_path}")
    print(f"Wrote: {json_path}")

if __name__ == "__main__":
    main()

