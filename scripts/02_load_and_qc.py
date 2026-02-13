#!/usr/bin/env python
"""
02_load_and_qc.py

Load dataset counts into AnnData, compute QC metrics and plots, filter conservatively,
and save work/qc/<label>.qc.h5ad.

This script tries, in order:
  1) existing *.h5ad
  2) Series-level processed long count matrix TSV.GZ (feature/cell/counts)
  3) GSM wide matrices (genes x cells in a .txt.gz per GSM) [optionally smoke-test]
  4) 10x / MTX

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
import gzip
import anndata as ad
from scipy import sparse
from pathlib import Path

from _plotting import set_publication_style, savefig_png_pdf
set_publication_style()


def protein_coding_mito_mask(varnames) -> np.ndarray:
    """
    Boolean mask selecting ONLY protein-coding mitochondrial genes:
      MT-ND*, MT-CO*, MT-ATP*, MT-CYB
    Excludes MT-RNR* and MT-T*.
    """
    s = pd.Index(varnames).astype(str)

    # uppercase once for consistent matching
    u = s.str.upper()

    mask = (
        u.str.startswith("MT-ND")
        | u.str.startswith("MT-CO")
        | u.str.startswith("MT-ATP")
        | (u == "MT-CYB")
    )

    # mask is a pandas Index/Series boolean array-like; normalize to numpy bool
    return np.asarray(mask, dtype=bool)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def deep_merge(a: dict, b: dict) -> dict:
    """Merge b into a (recursive), returning new dict."""
    out = dict(a)
    for k, v in (b or {}).items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str) -> dict:
    cfg = load_yaml(path)
    includes = cfg.pop("include", [])
    base = {}
    for inc in includes:
        inc_path = str(Path(path).parent / inc) if not str(inc).startswith("configs/") else inc
        base = deep_merge(base, load_yaml(inc_path))
    return deep_merge(base, cfg)


def find_first(patterns):
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None


def load_counts_from_wide_gsm_txt_gz(path: Path, keep_cells: set[str] | None = None) -> ad.AnnData:
    """
    GSM*.txt.gz format (wide):
      - first row: cell IDs (WMC...)
      - first col: gene names (no header label)
      - remaining: integer counts

    Returns AnnData with X shaped (cells x genes).
    """
    # Fast header parse to support column subsetting without reading whole file
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")

    # header contains ONLY cell IDs; gene column has no name (pandas index_col=0 handles it)
    if keep_cells is not None:
        wanted = [c for c in header if c in keep_cells]
        if not wanted:
            raise ValueError(f"No overlapping cells between file header and keep_cells for {path.name}")
        usecols = [0] + wanted  # 0 => gene column, then selected cell columns
        df = pd.read_csv(path, sep="\t", compression="gzip", index_col=0, usecols=usecols)
    else:
        df = pd.read_csv(path, sep="\t", compression="gzip", index_col=0)

    # df: genes x cells
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.int32)

    # Build AnnData: cells x genes
    X = sparse.csr_matrix(df.to_numpy(dtype=np.int32).T)
    a = ad.AnnData(X=X)
    a.obs_names = df.columns.astype(str)
    a.var_names = df.index.astype(str)
    a.obs_names_make_unique()
    a.var_names_make_unique()
    a.uns.setdefault("source_files", {})["gsm_wide_txt_gz"] = str(path)
    return a


def _load_from_mtx_or_10x(ds_dir: str) -> sc.AnnData:
    # Try 10x mtx layout
    mtx = find_first([
        os.path.join(ds_dir, "**", "matrix.mtx"),
        os.path.join(ds_dir, "**", "matrix.mtx.gz"),
        os.path.join(ds_dir, "**", "*matrix*.mtx"),
        os.path.join(ds_dir, "**", "*matrix*.mtx.gz"),
    ])

    # Also try 10x H5 (optional)
    h5_10x = find_first([
        os.path.join(ds_dir, "**", "*.h5"),
        os.path.join(ds_dir, "**", "*.h5.gz"),
    ])

    # Prefer MTX if present; otherwise try 10x h5
    if mtx:
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
            a = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", make_unique=True, cache=True)
            a.uns.setdefault("source_files", {})["10x_dir"] = mtx_dir
            return a

        # Fallback: read mtx and attempt to attach names found anywhere under ds_dir
        a = sc.read_mtx(mtx).T
        a.uns.setdefault("source_files", {})["mtx"] = mtx

        features2 = find_first([
            os.path.join(ds_dir, "**", "features.tsv*"),
            os.path.join(ds_dir, "**", "genes.tsv*"),
        ])
        barcodes2 = find_first([
            os.path.join(ds_dir, "**", "barcodes.tsv*"),
        ])

        if features2:
            feat_df = pd.read_csv(features2, sep="\t", header=None)
            if feat_df.shape[1] >= 2:
                a.var_names = feat_df.iloc[:, 1].astype(str).values
            else:
                a.var_names = feat_df.iloc[:, 0].astype(str).values
            a.var_names_make_unique()
            a.uns["source_files"]["features"] = features2

        if barcodes2:
            bc_df = pd.read_csv(barcodes2, sep="\t", header=None)
            a.obs_names = bc_df.iloc[:, 0].astype(str).values
            a.obs_names_make_unique()
            a.uns["source_files"]["barcodes"] = barcodes2

        return a

    if h5_10x:
        # Best-effort: only works for true 10x formatted h5
        try:
            a = sc.read_10x_h5(h5_10x)
            a.var_names_make_unique()
            a.obs_names_make_unique()
            a.uns.setdefault("source_files", {})["10x_h5"] = h5_10x
            return a
        except Exception:
            pass

    raise FileNotFoundError(f"No supported count files found under: {ds_dir}")


def load_counts_from_dir(ds_dir: str, qc_cfg: dict | None = None) -> sc.AnnData:
    qc_cfg = qc_cfg or {}

    # 1) Prefer existing h5ad if present
    h5ad = find_first([os.path.join(ds_dir, "**", "*.h5ad")])
    if h5ad:
        a = sc.read_h5ad(h5ad)
        a.uns.setdefault("source_files", {})["h5ad"] = h5ad
        return a

    # 2) Series-level processed count matrix (TSV.GZ), e.g. GSE210395_scRNA_countMatrix.tsv.gz
    tsv_gz = find_first([
        os.path.join(ds_dir, "**", "*countMatrix*.tsv.gz"),
        os.path.join(ds_dir, "**", "*countmatrix*.tsv.gz"),
        os.path.join(ds_dir, "**", "*count_matrix*.tsv.gz"),
    ])
    if tsv_gz:
        print(f"Found count matrix TSV.GZ (long format): {tsv_gz}")
        print("Parsing long table and building sparse matrix...")

        df = pd.read_csv(
            tsv_gz,
            sep="\t",
            compression="gzip",
            usecols=["feature", "cell", "counts"],
            dtype={"feature": "string", "cell": "string", "counts": "int32"},
        )

        genes = pd.Index(df["feature"].astype(str).unique(), name="gene")
        cells = pd.Index(df["cell"].astype(str).unique(), name="cell")

        gene_to_i = pd.Series(np.arange(len(genes), dtype=np.int32), index=genes)
        cell_to_j = pd.Series(np.arange(len(cells), dtype=np.int32), index=cells)

        row = gene_to_i[df["feature"].astype(str)].to_numpy(dtype=np.int32, copy=False)
        col = cell_to_j[df["cell"].astype(str)].to_numpy(dtype=np.int32, copy=False)
        data = df["counts"].to_numpy(dtype=np.float32, copy=False)

        X = sparse.coo_matrix((data, (col, row)), shape=(len(cells), len(genes))).tocsr()

        a = sc.AnnData(X=X)
        a.obs_names = cells.astype(str)
        a.var_names = genes.astype(str)
        a.obs_names_make_unique()
        a.var_names_make_unique()
        a.uns.setdefault("source_files", {})["countMatrix_tsv_gz_long"] = tsv_gz
        return a

    # 3) GSM wide matrices (e.g., GSE195452_RAW.tar extracted elsewhere; or present directly)
    gsm_files = sorted(Path(ds_dir).rglob("GSM*_*.txt.gz"))
    if gsm_files:
        # De-dupe by full relative path (handles same filename appearing in multiple dirs)
        gsm_files_unique = []
        seen = set()
        for p in gsm_files:
            rp = str(p.relative_to(ds_dir))  # unique within ds_dir even if filenames repeat
            if rp in seen:
                continue
            seen.add(rp)
            gsm_files_unique.append(p)

        # Smoke test / full load
        n = int(qc_cfg.get("gsm_smoke_n", 2))
        gsm_subset = gsm_files_unique if n == 0 else gsm_files_unique[:n]

        print(
            f"Found {len(gsm_files_unique)} GSM raw files (wide matrices); "
            f"loading {len(gsm_subset)} {'(all)' if n == 0 else 'for QC smoke test'}."
        )

        adatas = []
        keys = []
        for idx, p in enumerate(gsm_subset):
            a = load_counts_from_wide_gsm_txt_gz(p)

            # stable per-file identifier (unique even if GSM names repeat in different dirs)
            rel = str(p.relative_to(ds_dir))
            sample_id = rel.replace("/", "__")

            a.obs["sample_id"] = sample_id
            adatas.append(a)
            keys.append(sample_id)

        if len(adatas) == 1:
            out = adatas[0]
            out.obs_names_make_unique()
            return out

        out = ad.concat(
            adatas,
            axis=0,
            label="sample_id",
            keys=keys,          # now guaranteed unique
            merge="same",
            index_unique="-",   # prevent obs-name collisions across samples
        )
        out.obs_names_make_unique()
        return out

    # 4) MTX / 10x
    return _load_from_mtx_or_10x(ds_dir)


def qc_plots(adata: sc.AnnData, out_base_no_ext: str):
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    savefig_png_pdf(out_base_no_ext)


import re

def mito_mask(var_names, mt_prefixes=("MT-", "mt-")):
    v = np.asarray(var_names, dtype=str)

    # 1) explicit prefixes from config
    prefix_mask = np.zeros(len(v), dtype=bool)
    for p in mt_prefixes:
        prefix_mask |= np.char.startswith(v, p)

    # 2) common alternative human mito patterns
    #    MT-*, MT.*, mt-*, mt.*, MTND*, MTRNR*, MTCO*, MTCYB*, MTATP*, etc.
    rx = re.compile(r"^(MT[-\.]|mt[-\.]|MTND|MTRNR|MTCO|MTCYB|MTATP|mtnd|mtrnr|mtco|mtcyb|mtatp)")
    regex_mask = np.fromiter((bool(rx.match(x)) for x in v), dtype=bool, count=len(v))

    return prefix_mask | regex_mask


def compute_pct_counts_in_mask(adata, mask: np.ndarray, colname: str) -> None:
    """
    Compute percent of counts contributed by genes selected by `mask` and write to adata.obs[colname].
    Works for dense or sparse X. Produces a clean 1D float array. No divide warnings.
    """
    X = adata.X
    if sparse.issparse(X):
        total = np.asarray(X.sum(axis=1)).ravel().astype(np.float64, copy=False)
        mt = np.asarray(X[:, mask].sum(axis=1)).ravel().astype(np.float64, copy=False)
    else:
        total = X.sum(axis=1).astype(np.float64, copy=False)
        mt = X[:, mask].sum(axis=1).astype(np.float64, copy=False)

    pct = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(mt, total, out=pct, where=(total > 0))
    pct *= 100.0

    adata.obs[colname] = pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    datasets = cfg.get("datasets", [])
    if not datasets:
        raise SystemExit(f"No datasets found in config: {args.config} (after include merge)")

    os.makedirs(args.outdir, exist_ok=True)

    for ds in datasets:
        label = ds["label"]
        ds_dir = os.path.join(args.indir, label)
        print(f"\n=== QC: {label} ===")

        qc_cfg_global = cfg.get("qc", {}) or {}
        ds_overrides = (ds.get("overrides", {}) or {}).get("qc", {}) or {}
        qc_cfg = {**qc_cfg_global, **ds_overrides}


        adata = load_counts_from_dir(ds_dir, qc_cfg=qc_cfg)

        # -------------------------
        # Basic QC (robust version)
        # -------------------------

        # ---- 1. Drop zero-UMI cells immediately ----
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float32)

        before0 = adata.n_obs
        adata = adata[total_counts > 0].copy()
        after0 = adata.n_obs

        if after0 != before0:
            print(f"[QC] Dropped zero-UMI cells: {before0} -> {after0}")

        # Recompute totals after filtering
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float32)
        adata.obs["total_counts"] = total_counts

        # ---- 2. Compute genes per cell ----
        adata.obs["n_genes_by_counts"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

        # ---- 3. Build mitochondrial masks ----

        # Protein-coding MT genes (stable across pipelines)
        mt_pc_mask = protein_coding_mito_mask(adata.var_names)

        # All MT prefix genes (for reporting)
        var = np.asarray(adata.var_names, dtype=str)
        mt_all_mask = np.char.startswith(var, "MT-") | np.char.startswith(var, "mt-")


        print(f"MT genes (protein-coding only): {int(mt_pc_mask.sum())}")
        print(f"MT genes (all MT- prefix): {int(mt_all_mask.sum())}")

        # ---- 4. Compute mitochondrial percentages safely ----

        def compute_pct_mask(adata, mask):
            if mask.sum() == 0:
                return np.full(adata.n_obs, np.nan, dtype=np.float32)

            mt_counts = np.asarray(adata[:, mask].X.sum(axis=1)).ravel().astype(np.float32)
            total = adata.obs["total_counts"].values.astype(np.float32)

            pct = np.full_like(total, np.nan, dtype=np.float32)
            np.divide(mt_counts, total, out=pct, where=total > 0)
            pct *= 100.0
            return pct

        adata.obs["pct_counts_mt_pc"] = compute_pct_mask(adata, mt_pc_mask)
        adata.obs["pct_counts_mt_all"] = compute_pct_mask(adata, mt_all_mask)

        # Choose which column drives filtering
        mt_filter_mode = qc_cfg.get("mt_filter_mode", "protein_coding")
        if mt_filter_mode == "all":
            adata.obs["pct_counts_mt"] = adata.obs["pct_counts_mt_all"].astype(np.float32)
        else:
            adata.obs["pct_counts_mt"] = adata.obs["pct_counts_mt_pc"]

        # ---- 5. Print QC summary ----

        print(f"Cells: {adata.n_obs} Genes: {adata.n_vars}")
        print(f"Median genes/cell: {np.nanmedian(adata.obs['n_genes_by_counts'])}")
        print(f"Median UMIs/cell: {np.nanmedian(adata.obs['total_counts'])}")
        print(f"Median mt% (protein-coding): {np.nanmedian(adata.obs['pct_counts_mt_pc'])}")
        print(f"Median mt% (all MT): {np.nanmedian(adata.obs['pct_counts_mt_all'])}")

        # ---- 6. Write raw QC ----
        raw_out = os.path.join(args.outdir, f"{label}.raw_qc.h5ad")
        adata.write(raw_out)
        print(f"Wrote: {raw_out}")

        # ---- 7. Conservative filtering ----
        min_genes = int(qc_cfg.get("min_genes", 200))
        max_mt = float(qc_cfg.get("max_mt_pct", 25.0))

        before = adata.n_obs

        adata = adata[adata.obs["n_genes_by_counts"] >= min_genes].copy()
        adata = adata[adata.obs["pct_counts_mt"] <= max_mt].copy()

        after = adata.n_obs
        print(f"Filtered cells: {before} -> {after}")

        qc_out = os.path.join(args.outdir, f"{label}.qc.h5ad")
        adata.write(qc_out)
        print(f"Wrote: {qc_out}")


if __name__ == "__main__":
    main()

