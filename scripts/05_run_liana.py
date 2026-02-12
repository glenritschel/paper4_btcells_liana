#!/usr/bin/env python
"""
05_run_liana.py

Run LIANA receptor-ligand inference on B<->T interactions.

- Uses rank_aggregate.by_sample if sample_key exists and has >1 sample.
- Falls back to rank_aggregate if sample_key missing.

Writes:
  results/liana/<label>.liana.tsv
  results/liana/<label>.liana_top_bt.tsv
  results/liana/<label>.liana_top_tb.tsv

Usage:
  python scripts/05_run_liana.py --config configs/liana.yaml --indir work/scored --outdir results/liana
"""

import argparse
import os
import yaml
import pandas as pd
import scanpy as sc
import liana as li

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config).get("liana", {})
    groupby = cfg.get("groupby", "cell_type")
    sample_key = cfg.get("sample_key", "sample_id")
    use_raw = bool(cfg.get("use_raw", False))
    min_cells = int(cfg.get("min_cells_per_group", 50))
    sender = cfg.get("sender", "B_cell")
    receiver = cfg.get("receiver", "T_cell")

    os.makedirs(args.outdir, exist_ok=True)
    files = [f for f in os.listdir(args.indir) if f.endswith(".scored.h5ad")]
    if not files:
        raise SystemExit(f"No *.scored.h5ad in {args.indir}")

    for fn in sorted(files):
        label = fn.replace(".scored.h5ad", "")
        path = os.path.join(args.indir, fn)
        print(f"\n=== LIANA: {label} ===")
        adata = sc.read_h5ad(path)

        # Ensure groupby exists
        if groupby not in adata.obs.columns:
            raise RuntimeError(f"{groupby} not in adata.obs columns: {list(adata.obs.columns)}")

        # Subset to B and T only for speed/clarity
        adata_bt = adata[adata.obs[groupby].isin([sender, receiver])].copy()

        # Basic cell count gate
        counts = adata_bt.obs[groupby].value_counts()
        print("Group counts:\n", counts)
        if counts.get(sender, 0) < min_cells or counts.get(receiver, 0) < min_cells:
            print(f"WARNING: insufficient cells for LIANA (min {min_cells}). Skipping {label}.")
            continue

        # Choose per-sample mode if possible
        has_sample = sample_key in adata_bt.obs.columns and adata_bt.obs[sample_key].nunique() > 1
        try:
            if has_sample:
                print(f"Running rank_aggregate.by_sample(groupby={groupby}, sample_key={sample_key})")
                li.mt.rank_aggregate.by_sample(
                    adata_bt,
                    groupby=groupby,
                    sample_key=sample_key,
                    use_raw=use_raw,
                    verbose=True,
                )
            else:
                print(f"Running rank_aggregate(groupby={groupby}) (no usable {sample_key})")
                li.mt.rank_aggregate(
                    adata_bt,
                    groupby=groupby,
                    use_raw=use_raw,
                    verbose=True,
                )
        except Exception as e:
            raise RuntimeError(f"LIANA failed for {label}: {e}")

        res = adata_bt.uns.get("liana_res")
        if res is None:
            raise RuntimeError("No adata.uns['liana_res'] produced")

        # Persist full table
        out_all = os.path.join(args.outdir, f"{label}.liana.tsv")
        res.to_csv(out_all, sep="\t", index=False)
        print("Wrote:", out_all)

        # Filter key directions
        # LIANA columns typically: source, target, ligand_complex, receptor_complex, aggregate_rank, ...
        def top_direction(df, src, tgt, n=200):
            df2 = df[(df["source"] == src) & (df["target"] == tgt)].copy()
            # lower aggregate_rank is "better" in LIANA
            if "aggregate_rank" in df2.columns:
                df2 = df2.sort_values("aggregate_rank", ascending=True)
            return df2.head(n)

        bt = top_direction(res, sender, receiver, n=200)
        tb = top_direction(res, receiver, sender, n=200)

        out_bt = os.path.join(args.outdir, f"{label}.liana_top_bt.tsv")
        out_tb = os.path.join(args.outdir, f"{label}.liana_top_tb.tsv")
        bt.to_csv(out_bt, sep="\t", index=False)
        tb.to_csv(out_tb, sep="\t", index=False)
        print("Wrote:", out_bt)
        print("Wrote:", out_tb)

if __name__ == "__main__":
    main()

