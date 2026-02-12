#!/usr/bin/env python
"""
04_score_programs.py

Score signature/program gene sets from configs/scoring.yaml.
Adds one obs column per score key.

Usage:
  python scripts/04_score_programs.py --config configs/scoring.yaml --indir work/annot --outdir work/scored
"""

import argparse
import os
import yaml
import scanpy as sc

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def present(adata, genes):
    return [g for g in genes if g in adata.var_names]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    scores = cfg.get("scores", {})
    if not scores:
        raise SystemExit("No scores found in configs/scoring.yaml under key: scores")

    os.makedirs(args.outdir, exist_ok=True)
    files = [f for f in os.listdir(args.indir) if f.endswith(".annot.h5ad")]
    if not files:
        raise SystemExit(f"No *.annot.h5ad in {args.indir}")

    for fn in sorted(files):
        label = fn.replace(".annot.h5ad", "")
        path = os.path.join(args.indir, fn)
        print(f"\n=== Score programs: {label} ===")
        adata = sc.read_h5ad(path)

        for score_name, genes in scores.items():
            genes2 = present(adata, genes)
            if len(genes2) == 0:
                print(f"WARNING: score {score_name} has 0 genes present; writing zeros")
                adata.obs[score_name] = 0.0
                continue
            sc.tl.score_genes(adata, gene_list=genes2, score_name=score_name, use_raw=False)
            print(f"Scored {score_name}: {len(genes2)}/{len(genes)} genes present")

        out = os.path.join(args.outdir, f"{label}.scored.h5ad")
        adata.write(out)
        print("Wrote:", out)

if __name__ == "__main__":
    main()

