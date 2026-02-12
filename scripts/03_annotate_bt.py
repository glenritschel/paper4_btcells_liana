#!/usr/bin/env python
"""
03_annotate_bt.py

Annotate B vs T at minimum, plus:
- Plasma cell detection (MS4A1 low, MZB1/JCHAIN high)
- CD4 vs CD8 (IL7R/CCR7 vs NKG7/GZMB plus CD4/CD8A/CD8B)

Outputs work/annot/<label>.annot.h5ad with:
  obs["cell_type"] in {"B_cell","T_cell","Other"}
  obs["t_subtype"] in {"CD4_like","CD8_like","Tfh_like","Treg_like","Other"}
  obs["b_subtype"] in {"B_like","Plasma_like","Other"}

Usage:
  python scripts/03_annotate_bt.py --config configs/qc.yaml --indir work/qc --outdir work/annot
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import scanpy as sc

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def present(adata, genes):
    return [g for g in genes if g in adata.var_names]

def score(adata, genes, name):
    genes = present(adata, genes)
    if len(genes) == 0:
        adata.obs[name] = 0.0
        return
    sc.tl.score_genes(adata, gene_list=genes, score_name=name, use_raw=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ann_cfg = cfg.get("annotation", {})
    b_markers = ann_cfg.get("b_markers", ["CD19","MS4A1","CD79A"])
    t_markers = ann_cfg.get("t_markers", ["CD3D","CD3E","CD3G"])

    os.makedirs(args.outdir, exist_ok=True)
    files = [f for f in os.listdir(args.indir) if f.endswith(".qc.h5ad")]
    if not files:
        raise SystemExit(f"No *.qc.h5ad in {args.indir}")

    for fn in sorted(files):
        label = fn.replace(".qc.h5ad", "")
        path = os.path.join(args.indir, fn)
        print(f"\n=== Annotate: {label} ===")
        adata = sc.read_h5ad(path)

        # Core scores
        score(adata, b_markers, "B_score")
        score(adata, t_markers, "T_score")

        # Plasma markers (contamination/alternate B lineage)
        plasma = ["MZB1", "JCHAIN", "XBP1", "SDC1", "IGHG1", "IGHG3", "IGKC"]
        score(adata, plasma, "Plasma_score")

        # CD4/CD8/Tfh/Treg heuristics (fast, not deep clustering)
        cd4_like = ["CD4", "IL7R", "CCR7", "LTB", "MALAT1"]
        cd8_like = ["CD8A", "CD8B", "NKG7", "GZMB", "PRF1", "GNLY"]
        tfh_like = ["BCL6", "CXCR5", "PDCD1", "ICOS", "IL21"]
        treg_like = ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT"]

        score(adata, cd4_like, "CD4_like_score")
        score(adata, cd8_like, "CD8_like_score")
        score(adata, tfh_like, "Tfh_like_score")
        score(adata, treg_like, "Treg_like_score")

        # Provisional labeling thresholds (tune later; start permissive)
        adata.obs["cell_type"] = "Other"
        adata.obs.loc[adata.obs["B_score"] > 0.3, "cell_type"] = "B_cell"
        adata.obs.loc[adata.obs["T_score"] > 0.3, "cell_type"] = "T_cell"

        # B subtype
        adata.obs["b_subtype"] = "Other"
        adata.obs.loc[adata.obs["cell_type"] == "B_cell", "b_subtype"] = "B_like"
        # Plasma override: if plasma high and MS4A1 low-ish, mark plasma-like
        if "MS4A1" in adata.var_names:
            ms4a1 = np.asarray(adata[:, "MS4A1"].X).reshape(-1)
        else:
            ms4a1 = np.zeros(adata.n_obs)

        plasma_hi = (adata.obs["Plasma_score"] > 0.3) & (ms4a1 < np.quantile(ms4a1, 0.6))
        adata.obs.loc[plasma_hi, "b_subtype"] = "Plasma_like"

        # T subtype
        adata.obs["t_subtype"] = "Other"
        is_t = adata.obs["cell_type"] == "T_cell"
        # pick best of CD4/CD8 unless tfh/treg dominate
        adata.obs.loc[is_t & (adata.obs["CD4_like_score"] >= adata.obs["CD8_like_score"]), "t_subtype"] = "CD4_like"
        adata.obs.loc[is_t & (adata.obs["CD8_like_score"] > adata.obs["CD4_like_score"]), "t_subtype"] = "CD8_like"
        adata.obs.loc[is_t & (adata.obs["Tfh_like_score"] > 0.3), "t_subtype"] = "Tfh_like"
        adata.obs.loc[is_t & (adata.obs["Treg_like_score"] > 0.3), "t_subtype"] = "Treg_like"

        # Save quick counts
        print("cell_type counts:\n", adata.obs["cell_type"].value_counts())
        print("b_subtype counts:\n", adata.obs["b_subtype"].value_counts())
        print("t_subtype counts:\n", adata.obs["t_subtype"].value_counts())

        out = os.path.join(args.outdir, f"{label}.annot.h5ad")
        adata.write(out)
        print("Wrote:", out)

if __name__ == "__main__":
    main()

