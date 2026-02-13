#!/usr/bin/env python3
"""
Create within-dataset B_low / B_high subsets (for datasets with a single sample_id).
Keeps all T cells, splits B cells by quartiles of a score column.

Outputs:
  <label>.BT.Blow.scored.h5ad
  <label>.BT.Bhigh.scored.h5ad
  <label>.within_dataset_split.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import scanpy as sc


@dataclass
class SplitMeta:
    input_h5ad: str
    label: str
    groupby: str
    b_label: str
    t_label: str
    score_col: str
    q_low: float
    q_high: float
    n_total: int
    n_bt: int
    n_b: int
    n_t: int
    n_b_low: int
    n_b_high: int
    out_blow: str
    out_bhigh: str


def _quantiles(x: np.ndarray, qlo: float, qhi: float) -> Tuple[float, float]:
    x = x.astype(float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise RuntimeError("All values are NaN; cannot compute quantiles.")
    return float(np.quantile(x, qlo)), float(np.quantile(x, qhi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .scored.h5ad")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--label", default=None, help="Label prefix (default: basename without .scored.h5ad/.h5ad)")
    ap.add_argument("--groupby", default="cell_type", help="Obs column defining cell types (default: cell_type)")
    ap.add_argument("--b-label", default="B_cell", help="Value in groupby representing B cells")
    ap.add_argument("--t-label", default="T_cell", help="Value in groupby representing T cells")
    ap.add_argument("--score-col", required=True, help="Obs column used to split B cells (e.g., b_costim)")
    ap.add_argument("--qlo", type=float, default=0.25, help="Lower quantile (default 0.25)")
    ap.add_argument("--qhi", type=float, default=0.75, help="Upper quantile (default 0.75)")
    ap.add_argument("--min-b", type=int, default=50, help="Min B cells required in each subset (default 50)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base = os.path.basename(args.input)
    if args.label:
        label = args.label
    else:
        label = base.replace(".scored.h5ad", "").replace(".h5ad", "")

    ad = sc.read_h5ad(args.input)
    if args.groupby not in ad.obs:
        raise RuntimeError(f"Missing obs['{args.groupby}']. Have: {list(ad.obs.columns)[:50]}")

    # Restrict to B+T only
    bt = ad[ad.obs[args.groupby].isin([args.b_label, args.t_label])].copy()
    if bt.n_obs == 0:
        raise RuntimeError("No B/T cells after subsetting.")

    if args.score_col not in bt.obs:
        raise RuntimeError(f"Missing obs['{args.score_col}']. Have: {list(bt.obs.columns)[:80]}")

    b = bt[bt.obs[args.groupby] == args.b_label].copy()
    t = bt[bt.obs[args.groupby] == args.t_label].copy()

    q_low, q_high = _quantiles(b.obs[args.score_col].to_numpy(), args.qlo, args.qhi)

    # Label B cells and build subsets: (B_low + all T) and (B_high + all T)
    bt.obs["b_bin"] = "mid"
    bt.obs.loc[(bt.obs[args.groupby] == args.b_label) & (bt.obs[args.score_col].astype(float) <= q_low), "b_bin"] = "B_low"
    bt.obs.loc[(bt.obs[args.groupby] == args.b_label) & (bt.obs[args.score_col].astype(float) >= q_high), "b_bin"] = "B_high"

    blow = bt[(bt.obs[args.groupby] == args.t_label) | (bt.obs["b_bin"] == "B_low")].copy()
    bhigh = bt[(bt.obs[args.groupby] == args.t_label) | (bt.obs["b_bin"] == "B_high")].copy()

    n_b_low = int((bt.obs["b_bin"] == "B_low").sum())
    n_b_high = int((bt.obs["b_bin"] == "B_high").sum())

    if n_b_low < args.min_b or n_b_high < args.min_b:
        raise RuntimeError(
            f"Too few B cells in a split: B_low={n_b_low} B_high={n_b_high} (min_b={args.min_b}). "
            f"Try lowering --min-b or changing quantiles."
        )

    out_blow = os.path.join(args.outdir, f"{label}.BT.Blow.scored.h5ad")
    out_bhigh = os.path.join(args.outdir, f"{label}.BT.Bhigh.scored.h5ad")
    meta_path = os.path.join(args.outdir, f"{label}.within_dataset_split.json")

    blow.write(out_blow)
    bhigh.write(out_bhigh)

    meta = SplitMeta(
        input_h5ad=args.input,
        label=label,
        groupby=args.groupby,
        b_label=args.b_label,
        t_label=args.t_label,
        score_col=args.score_col,
        q_low=q_low,
        q_high=q_high,
        n_total=int(ad.n_obs),
        n_bt=int(bt.n_obs),
        n_b=int(b.n_obs),
        n_t=int(t.n_obs),
        n_b_low=n_b_low,
        n_b_high=n_b_high,
        out_blow=out_blow,
        out_bhigh=out_bhigh,
    )
    with open(meta_path, "w") as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"[OK] wrote {out_blow} (cells={blow.n_obs})")
    print(f"[OK] wrote {out_bhigh} (cells={bhigh.n_obs})")
    print(f"[OK] wrote {meta_path}")
    print(f"[INFO] B quantiles {args.qlo:.2f}/{args.qhi:.2f}: {q_low:.6g} / {q_high:.6g}")
    print(f"[INFO] split sizes: B_low={n_b_low} B_high={n_b_high}  T(all)={t.n_obs}")


if __name__ == "__main__":
    main()

