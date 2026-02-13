#!/usr/bin/env python3
"""
Compute ligand/receptor expression amplification between two conditions on shared LR pairs.

Inputs:
  --adata-a: AnnData for condition A (e.g., Q1)
  --adata-b: AnnData for condition B (e.g., Q4)
  --overlap: overlap_by_delta.tsv from 07_compare_liana_datasets.py (shared LR pairs)
  --outdir:  output directory

Outputs:
  - *.lr_amplification.tsv   (per LR pair ligand/receptor means + log2FC + pair amplification)
  - *.lr_amplification.summary.json (summary stats + tests)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats


def _safe_mean_expr(adata: sc.AnnData, gene: str) -> float:
    """Mean expression of a single gene across all cells in adata (uses .X)."""
    if gene not in adata.var_names:
        return float("nan")
    X = adata[:, [gene]].X  # shape (n,1)
    # robust conversion for sparse views
    if hasattr(X, "toarray"):
        arr = X.toarray()
    else:
        arr = np.asarray(X)
    return float(np.nanmean(arr))


def _log2fc(b: float, a: float, eps: float = 1e-6) -> float:
    """log2((b+eps)/(a+eps))"""
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return float(np.log2((b + eps) / (a + eps)))


@dataclass
class Cfg:
    adata_a: str
    adata_b: str
    overlap_tsv: str
    outdir: str
    label_a: str
    label_b: str
    sender: str = "B_cell"
    receiver: str = "T_cell"
    focus: str = "BtoT"  # BtoT only, or ALL


def parse_args() -> Cfg:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata-a", required=True, help="AnnData for condition A (e.g., Q1)")
    ap.add_argument("--adata-b", required=True, help="AnnData for condition B (e.g., Q4)")
    ap.add_argument("--overlap", required=True, help="overlap_by_delta.tsv from 07_compare_liana_datasets.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--sender", default="B_cell")
    ap.add_argument("--receiver", default="T_cell")
    ap.add_argument("--focus", choices=["BtoT", "ALL"], default="BtoT",
                    help="Restrict to B->T pairs (recommended) or compute all directions")
    args = ap.parse_args()
    return Cfg(
        adata_a=args.adata_a,
        adata_b=args.adata_b,
        overlap_tsv=args.overlap,
        outdir=args.outdir,
        label_a=args.label_a,
        label_b=args.label_b,
        sender=args.sender,
        receiver=args.receiver,
        focus=args.focus,
    )


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)

    # Load LR overlap table (shared pairs; includes rank_a/rank_b/delta_rank)
    ov = pd.read_csv(cfg.overlap_tsv, sep="\t")
    required = {"source", "target", "ligand_complex", "receptor_complex", "rank_a", "rank_b"}
    missing = required - set(ov.columns)
    if missing:
        raise RuntimeError(f"overlap file missing columns: {sorted(missing)}")

    if cfg.focus == "BtoT":
        ov = ov[(ov["source"] == cfg.sender) & (ov["target"] == cfg.receiver)].copy()

    if ov.empty:
        raise RuntimeError("No LR pairs remain after filtering (check --focus/--sender/--receiver).")

    # Load AnnData
    ad_a = sc.read_h5ad(cfg.adata_a)
    ad_b = sc.read_h5ad(cfg.adata_b)

    # Subset to BT only (consistent with your LIANA)
    bt_keep = {cfg.sender, cfg.receiver}
    if "cell_type" not in ad_a.obs or "cell_type" not in ad_b.obs:
        raise RuntimeError("Expected obs['cell_type'] in both AnnData files.")
    ad_a = ad_a[ad_a.obs["cell_type"].isin(bt_keep)].copy()
    ad_b = ad_b[ad_b.obs["cell_type"].isin(bt_keep)].copy()

    # Split by sender/receiver
    a_sender = ad_a[ad_a.obs["cell_type"] == cfg.sender].copy()
    a_recv = ad_a[ad_a.obs["cell_type"] == cfg.receiver].copy()
    b_sender = ad_b[ad_b.obs["cell_type"] == cfg.sender].copy()
    b_recv = ad_b[ad_b.obs["cell_type"] == cfg.receiver].copy()

    rows = []
    for _, r in ov.iterrows():
        lig = str(r["ligand_complex"])
        rec = str(r["receptor_complex"])

        lig_a = _safe_mean_expr(a_sender, lig)
        lig_b = _safe_mean_expr(b_sender, lig)
        rec_a = _safe_mean_expr(a_recv, rec)
        rec_b = _safe_mean_expr(b_recv, rec)

        lig_l2fc = _log2fc(lig_b, lig_a)
        rec_l2fc = _log2fc(rec_b, rec_a)

        # Pair amplification index:
        # average of ligand and receptor log2FC (simple, interpretable)
        if np.isnan(lig_l2fc) and np.isnan(rec_l2fc):
            pair_amp = float("nan")
        elif np.isnan(rec_l2fc):
            pair_amp = lig_l2fc
        elif np.isnan(lig_l2fc):
            pair_amp = rec_l2fc
        else:
            pair_amp = 0.5 * (lig_l2fc + rec_l2fc)

        rows.append({
            "lr_id": r.get("lr_id", f"{r['source']}|{r['target']}|{lig}|{rec}"),
            "source": r["source"],
            "target": r["target"],
            "ligand_complex": lig,
            "receptor_complex": rec,
            "rank_a": float(r["rank_a"]),
            "rank_b": float(r["rank_b"]),
            # keep delta_rank ONLY if present in input (your delta file has it)
            "delta_rank": float(r["delta_rank"]) if "delta_rank" in ov.columns else (float(r["rank_b"]) - float(r["rank_a"])),
            "lig_mean_a": lig_a,
            "lig_mean_b": lig_b,
            "lig_log2fc": lig_l2fc,
            "rec_mean_a": rec_a,
            "rec_mean_b": rec_b,
            "rec_log2fc": rec_l2fc,
            "pair_amp_log2fc": pair_amp,
            "abs_pair_amp_log2fc": float(abs(pair_amp)) if not np.isnan(pair_amp) else float("nan"),
        })

    out = pd.DataFrame(rows)

    # Sort for mechanistic selection: biggest expression amplification first
    out = out.sort_values(["abs_pair_amp_log2fc", "abs_delta_rank" if "abs_delta_rank" in out.columns else "rank_a"],
                          ascending=[False, False], na_position="last")

    tag = f"{cfg.label_a}_vs_{cfg.label_b}"
    out_tsv = os.path.join(cfg.outdir, f"lr_amplification_{tag}.tsv")
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] Wrote: {out_tsv}")

    # ---- Summary stats + tests (ligand, receptor, pair_amp) ----
    def _finite(x: pd.Series) -> np.ndarray:
        v = x.to_numpy(dtype=float)
        return v[np.isfinite(v)]

    lig_fc = _finite(out["lig_log2fc"])
    rec_fc = _finite(out["rec_log2fc"])
    pair_fc = _finite(out["pair_amp_log2fc"])

    # Directional “amplification” evidence: are these > 0?
    def _binom_pos(v: np.ndarray) -> dict:
        if v.size == 0:
            return {"n": 0}
        k = int((v > 0).sum())
        n = int(v.size)
        # one-sided: P(X>=k | p=0.5)
        p = float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)
        return {"n": n, "n_pos": k, "frac_pos": k / n, "binom_p_greater_0": p}

    summary = {
        "labels": {"a": cfg.label_a, "b": cfg.label_b},
        "inputs": {"adata_a": cfg.adata_a, "adata_b": cfg.adata_b, "overlap": cfg.overlap_tsv, "focus": cfg.focus},
        "n_pairs": int(len(out)),
        "log2fc_stats": {
            "ligand": {
                "median": float(np.nanmedian(lig_fc)) if lig_fc.size else None,
                "mean": float(np.nanmean(lig_fc)) if lig_fc.size else None,
                "pos_test": _binom_pos(lig_fc),
            },
            "receptor": {
                "median": float(np.nanmedian(rec_fc)) if rec_fc.size else None,
                "mean": float(np.nanmean(rec_fc)) if rec_fc.size else None,
                "pos_test": _binom_pos(rec_fc),
            },
            "pair_amp": {
                "median": float(np.nanmedian(pair_fc)) if pair_fc.size else None,
                "mean": float(np.nanmean(pair_fc)) if pair_fc.size else None,
                "pos_test": _binom_pos(pair_fc),
            },
        },
    }

    out_json = os.path.join(cfg.outdir, f"lr_amplification_{tag}.summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Wrote: {out_json}")


if __name__ == "__main__":
    main()

