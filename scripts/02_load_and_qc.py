#!/usr/bin/env python
"""
02_load_and_qc.py

Load dataset counts into AnnData, compute QC metrics and plots, filter conservatively,
and save work/qc/<label>.qc.h5ad.

This script tries, in order:
  1) 10x directory structure (matrix.mtx + features.tsv + barcodes.tsv)
  2) MTX alone (matrix.mtx) with optional features/barcodes if found
  3) h5ad if present (pass-through QC)

Usage:
  python scripts/02_load_and_qc.py --config configs/qc.yaml --indir work/geo --outdir work/qc
"""

import argparse
import os
import glob
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from _plotting import set_publication_style
set_publication_style()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_first(patterns):
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None

def load_counts_from_dir(ds_dir: str) -> sc.AnnData:
    # Prefer existing h5ad if present
    h5ad = find_first([os.path.join(ds_dir, "**", "*.h5ad")])
    if h5ad:
        adata = sc.read_h5ad(h5ad)
        adata.uns.setdefault("source_files", {})["h5ad"] = h5ad
        return adata

    # Try 10x mtx layout
    mtx = find_first([
        os.path.join(ds_dir, "**", "matrix.mtx"),
        os.path.join(ds_dir, "**", "matrix.mtx.gz"),
        os.path.join(ds_dir, "**", "*matrix*.mtx"),
        os.path.join(ds_dir, "**", "*matrix*.mtx.gz"),
    ])
    if not mtx:
        raise FileNotFoundError(f"No matrix file found under: {ds_dir}")

    # If 10x structure exists, use read_10x_mtx
    mtx_dir = os.path.dirname(mtx)
    features = find_first([
        os.path.join(mtx_dir, "features.tsv"),
        os.path.join(mtx_dir, "features.tsv.gz"),
        os.path.join(mtx_dir, "genes.tsv"),
        os.path.join(mtx_dir, "genes.tsv.gz"),
    ])
    barcodes = find_first([
        os.path.join(mtx_dir, "barcodes.tsv"),
        os.path.join(mtx_dir, "barcodes.tsv.gz"),
    ])

    if features and barcodes:
        adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", make_unique=True, cache=True)
        adata.uns.setdefault("source_files", {})["10x_dir"] = mtx_dir
        return adata

    # Fallback: read mtx and attempt to attach names if present anywhere nearby
    adata = sc.read_mtx(mtx).T
    adata.uns.setdefault("source_files", {})["mtx"] = mtx

    # Try locate features/barcodes anywhere under ds_dir (not perfect, but helps)
    features2 = find_first([
        os.path.join(ds_dir, "**", "features.tsv*"),
        os.path.join(ds_dir, "**", "genes.tsv*"),
    ])
    barcodes2 = find_first([
        os.path.join(ds_dir, "**", "barcodes.tsv*"),
    ])

    if features2:
        feat_df = pd.read_csv(features2, sep="\t", header=None)
        # 10x: col1=gene_id, col2=gene_name; genes.tsv: col0=gene_id, col1=gene_name
        if feat_df.shape[1] >= 2:
            adata.var_names = feat_df.iloc[:, 1].astype(str).values
        else:
            adata.var_names = feat_df.iloc[:, 0].astype(str).values
        adata.var_names_make_unique()
        adata.uns["source_files"]["features"] = features2

    if barcodes2:
        bc_df = pd.read_csv(barcodes2, sep="\t", header=None)
        adata.obs_names = bc_df.iloc[:, 0].astype(str).values
        adata.obs_names_make_unique()
        adata.uns["source_files"]["barcodes"] = barcodes2

    return adata

def qc_plots(adata: sc.AnnData, out_png: str):
    plt.figure()
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--indir", required=True, help="work/geo")
    ap.add_argument("--outdir", required=True, help="work/qc")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    qc_cfg = cfg.get("qc", {})
    mt_prefixes = tuple(qc_cfg.get("mt_prefixes", ["MT-", "mt-"]))
    min_genes = int(qc_cfg.get("min_genes", 200))
    max_genes = int(qc_cfg.get("max_genes", 6000))
    max_mt = float(qc_cfg.get("max_mt_pct", 15))

    os.makedirs(args.outdir, exist_ok=True)

    # Each dataset in indir is a label directory
    labels = sorted([d for d in os.listdir(args.indir) if os.path.isdir(os.path.join(args.indir, d))])
    if not labels:
        raise SystemExit(f"No datasets found under {args.indir}")

    for label in labels:
        ds_dir = os.path.join(args.indir, label)
        print(f"\n=== QC: {label} ===")
        adata = load_counts_from_dir(ds_dir)

        # Minimal sanity
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise RuntimeError(f"Empty AnnData loaded for {label}")

        # Mito gene flag
        adata.var["mt"] = adata.var_names.astype(str).str.startswith(mt_prefixes)

        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

        print("Cells:", adata.n_obs, "Genes:", adata.n_vars)
        print("Median genes/cell:", float(np.median(adata.obs["n_genes_by_counts"])))
        print("Median UMIs/cell:", float(np.median(adata.obs["total_counts"])))
        print("Median mt%:", float(np.median(adata.obs["pct_counts_mt"])))

        # Save pre-filter metrics
        pre_path = os.path.join(args.outdir, f"{label}.raw_qc.h5ad")
        adata.write(pre_path)
        print("Wrote:", pre_path)

        # Plot pre-filter
        qc_plots(adata, os.path.join(args.outdir, f"{label}.qc_violin_pre.png"))

        # Conservative filters
        keep = (
            (adata.obs["n_genes_by_counts"] >= min_genes) &
            (adata.obs["n_genes_by_counts"] <= max_genes) &
            (adata.obs["pct_counts_mt"] <= max_mt)
        )
        before = adata.n_obs
        adata = adata[keep].copy()
        after = adata.n_obs
        print(f"Filtered cells: {before} -> {after}")

        # Normalize/log for downstream scoring/annotation
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Plot post-filter
        qc_plots(adata, os.path.join(args.outdir, f"{label}.qc_violin_post.png"))

        out_path = os.path.join(args.outdir, f"{label}.qc.h5ad")
        adata.write(out_path)
        print("Wrote:", out_path)

if __name__ == "__main__":
    main()

