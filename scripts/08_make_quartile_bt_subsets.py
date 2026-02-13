#!/usr/bin/env python
"""
08_make_quartile_bt_subsets.py

Create BT subsets by sample-level stratification of a score (e.g. b_activation).
Default: bottom quartile (Q1) vs top quartile (Q4) based on per-sample mean score in B cells.

Inputs:
  - *.scored.h5ad (expects obs: cell_type, sample_id, and the score column)

Outputs:
  - {label}.BT.q1.h5ad
  - {label}.BT.q4.h5ad
  - {label}.per_sample_scores.tsv
  - {label}.quartiles.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc


def _find_score_col(obs: pd.DataFrame, requested: str) -> Optional[str]:
    """Find score column robustly: exact, case-insensitive, and common prefixes."""
    if requested in obs.columns:
        return requested

    req_norm = requested.strip().lower()

    # Case-insensitive exact match
    for c in obs.columns:
        if str(c).strip().lower() == req_norm:
            return c

    # Common prefixes people accidentally introduce
    candidates = []
    for c in obs.columns:
        cn = str(c).strip().lower()
        if cn in {f"score_{req_norm}", f"{req_norm}_score", f"{req_norm}_genescore"}:
            candidates.append(c)

    if len(candidates) == 1:
        return candidates[0]

    # Partial contains (last resort, but deterministic)
    contains = [c for c in obs.columns if req_norm in str(c).strip().lower()]
    if len(contains) == 1:
        return contains[0]

    return None


def _require_cols(label: str, adata: sc.AnnData, cols: list[str]) -> None:
    missing = [c for c in cols if c not in adata.obs.columns]
    if missing:
        raise SystemExit(
            f"[ERROR] {label}: missing obs columns: {missing}. "
            f"Available obs columns: {list(adata.obs.columns)[:50]}"
        )


@dataclass(frozen=True)
class SplitConfig:
    groupby: str
    sample_key: str
    sender: str
    receiver: str
    score_col: str
    min_b_cells: int
    min_t_cells: int
    mode: str  # 'quartiles' or 'median'
    q_low: float
    q_high: float


def _compute_per_sample_table(
    adata: sc.AnnData, cfg: SplitConfig
) -> pd.DataFrame:
    # BT-only view for counting
    bt = adata[adata.obs[cfg.groupby].isin([cfg.sender, cfg.receiver])].copy()

    # Per-sample counts for B and T
    counts = (
        bt.obs.groupby([cfg.sample_key, cfg.groupby], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
    )
    for c in [cfg.sender, cfg.receiver]:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[[cfg.sender, cfg.receiver]].rename(
        columns={cfg.sender: "n_B", cfg.receiver: "n_T"}
    )

    # Score aggregation uses B cells only
    b = adata[adata.obs[cfg.groupby] == cfg.sender].copy()
    # Mean score per sample (B cells)
    score = b.obs.groupby(cfg.sample_key)[cfg.score_col].mean().rename("b_score_mean")

    out = counts.join(score, how="left").reset_index().rename(columns={cfg.sample_key: "sample_id"})
    out["eligible"] = (out["n_B"] >= cfg.min_b_cells) & (out["n_T"] >= cfg.min_t_cells) & out["b_score_mean"].notna()
    return out


def _choose_samples(per_sample: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.Series, dict]:
    elig = per_sample[per_sample["eligible"]].copy()
    if len(elig) < 4:
        raise SystemExit(
            f"[ERROR] Not enough eligible samples to split. eligible={len(elig)}. "
            f"Try lowering min_b_cells/min_t_cells."
        )

    s = elig["b_score_mean"].astype(float)

    if cfg.mode == "median":
        thr = float(np.nanmedian(s))
        q1_ids = elig.loc[s <= thr, "sample_id"]
        q4_ids = elig.loc[s >= thr, "sample_id"]
        meta = {"mode": "median", "threshold": thr}
        return q1_ids, {"q4_samples": q4_ids, **meta}

    # quartiles
    qlo = float(np.nanquantile(s, cfg.q_low))
    qhi = float(np.nanquantile(s, cfg.q_high))
    q1_ids = elig.loc[s <= qlo, "sample_id"]
    q4_ids = elig.loc[s >= qhi, "sample_id"]

    meta = {"mode": "quartiles", "q_low": cfg.q_low, "q_high": cfg.q_high, "q1_threshold": qlo, "q4_threshold": qhi}
    return q1_ids, {"q4_samples": q4_ids, **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory containing *.scored.h5ad")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--score-col", required=True, help="obs column to stratify on (e.g., b_activation)")
    ap.add_argument("--groupby", default="cell_type")
    ap.add_argument("--sample-key", default="sample_id")
    ap.add_argument("--sender", default="B_cell")
    ap.add_argument("--receiver", default="T_cell")

    ap.add_argument("--min-b-cells", type=int, default=50)
    ap.add_argument("--min-t-cells", type=int, default=50)

    ap.add_argument("--mode", choices=["quartiles", "median"], default="quartiles")
    ap.add_argument("--q-low", type=float, default=0.25)
    ap.add_argument("--q-high", type=float, default=0.75)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.indir) if f.endswith(".scored.h5ad")])
    if not files:
        raise SystemExit(f"[ERROR] No *.scored.h5ad found in: {args.indir}")

    for fn in files:
        label = fn.replace(".scored.h5ad", "")
        path = os.path.join(args.indir, fn)
        print(f"\n=== Quartiles: {label} ===")
        adata = sc.read_h5ad(path)

        _require_cols(label, adata, [args.groupby, args.sample_key])

        # Find score column robustly
        score_col = _find_score_col(adata.obs, args.score_col)
        if score_col is None:
            # Print useful candidates and fail fast
            candidates = [c for c in adata.obs.columns if "score" in str(c).lower() or args.score_col.lower() in str(c).lower()]
            print(f"[ERROR] {label}: missing obs['{args.score_col}'] in {path}")
            print("        Available obs cols (first 60):", list(adata.obs.columns)[:60])
            if candidates:
                print("        Candidate score-like cols:", candidates[:40])
            raise SystemExit(2)

        cfg = SplitConfig(
            groupby=args.groupby,
            sample_key=args.sample_key,
            sender=args.sender,
            receiver=args.receiver,
            score_col=score_col,
            min_b_cells=args.min_b_cells,
            min_t_cells=args.min_t_cells,
            mode=args.mode,
            q_low=args.q_low,
            q_high=args.q_high,
        )

        per_sample = _compute_per_sample_table(adata, cfg)
        per_sample_out = os.path.join(args.outdir, f"{label}.per_sample_scores.tsv")
        per_sample.to_csv(per_sample_out, sep="\t", index=False)
        print("[OK] Wrote:", per_sample_out)

        q1_ids, meta = _choose_samples(per_sample, cfg)
        q4_ids = meta["q4_samples"]
        meta.pop("q4_samples")

        # Subset to BT cells and selected samples
        bt = adata[adata.obs[cfg.groupby].isin([cfg.sender, cfg.receiver])].copy()

        q1 = bt[bt.obs[cfg.sample_key].isin(set(q1_ids))].copy()
        q4 = bt[bt.obs[cfg.sample_key].isin(set(q4_ids))].copy()

        # Safety: ensure both groups have both cell types
        def _counts(x: sc.AnnData) -> dict:
            vc = x.obs[cfg.groupby].value_counts().to_dict()
            return {cfg.sender: int(vc.get(cfg.sender, 0)), cfg.receiver: int(vc.get(cfg.receiver, 0))}

        q1c, q4c = _counts(q1), _counts(q4)
        print(f"[INFO] Q1 samples={len(set(q1_ids))} BT cells={q1.n_obs} counts={q1c}")
        print(f"[INFO] Q4 samples={len(set(q4_ids))} BT cells={q4.n_obs} counts={q4c}")

        q1_out = os.path.join(args.outdir, f"{label}.BT.q1.h5ad")
        q4_out = os.path.join(args.outdir, f"{label}.BT.q4.h5ad")
        q1.write(q1_out)
        q4.write(q4_out)
        print("[OK] Wrote:", q1_out)
        print("[OK] Wrote:", q4_out)

        meta_out = os.path.join(args.outdir, f"{label}.quartiles.json")
        payload = {
            "label": label,
            "input": path,
            "mode": cfg.mode,
            "groupby": cfg.groupby,
            "sample_key": cfg.sample_key,
            "sender": cfg.sender,
            "receiver": cfg.receiver,
            "score_col_requested": args.score_col,
            "score_col_used": cfg.score_col,
            "min_b_cells": cfg.min_b_cells,
            "min_t_cells": cfg.min_t_cells,
            "n_samples_total": int(per_sample.shape[0]),
            "n_samples_eligible": int(per_sample["eligible"].sum()),
            "n_samples_q1": int(len(set(q1_ids))),
            "n_samples_q4": int(len(set(q4_ids))),
            "bt_cells_q1": int(q1.n_obs),
            "bt_cells_q4": int(q4.n_obs),
            "bt_counts_q1": q1c,
            "bt_counts_q4": q4c,
            **meta,
        }
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("[OK] Wrote:", meta_out)


if __name__ == "__main__":
    main()

