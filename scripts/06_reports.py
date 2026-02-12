#!/usr/bin/env python
"""
06_reports.py

Generate a minimal report:
- Summary counts (B/T counts if available from earlier h5ad)
- Top interactions focusing on your hypotheses (MHC-II, CD80/CD86-CD28, cytokines)

Usage:
  python scripts/06_reports.py --indir results/liana --outdir results
"""

import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from _plotting import set_publication_style
set_publication_style()

FOCUS_PAIRS = [
    # Costimulation
    ("CD80", "CD28"),
    ("CD86", "CD28"),
    # CD40 axis
    ("CD40", "CD40LG"),
    ("ICOSLG", "ICOS"),
    # CXCL13 axis (Tfh)
    ("CXCL13", "CXCR5"),
    # Cytokines of interest
    ("IL6", "IL6R"),
    ("TGFB1", "TGFBR1"),
    ("TGFB1", "TGFBR2"),
    ("IL1B", "IL1R1"),
    ("IL4", "IL4R"),
    ("IL13", "IL13RA1"),
    ("IL21", "IL21R"),
]

MHC_PREFIXES = ("HLA-D",)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="results/liana")
    ap.add_argument("--outdir", required=True, help="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_tables = os.path.join(args.outdir, "tables")
    out_figs = os.path.join(args.outdir, "figures")
    os.makedirs(out_tables, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.indir, "*.liana.tsv")))
    if not files:
        raise SystemExit(f"No *.liana.tsv in {args.indir}")

    summary_rows = []
    focus_rows = []

    for f in files:
        label = os.path.basename(f).replace(".liana.tsv", "")
        df = pd.read_csv(f, sep="\t")
        summary_rows.append({
            "label": label,
            "n_rows": len(df),
            "n_sources": df["source"].nunique() if "source" in df.columns else None,
            "n_targets": df["target"].nunique() if "target" in df.columns else None,
        })

        # Normalize column names we care about
        ligand_col = "ligand_complex" if "ligand_complex" in df.columns else "ligand"
        receptor_col = "receptor_complex" if "receptor_complex" in df.columns else "receptor"

        if ligand_col not in df.columns or receptor_col not in df.columns:
            continue

        # Focus set: exact ligand/receptor matches (complex names can include underscores)
        def contains_pair(lig, rec, lig_gene, rec_gene):
            return (lig_gene in str(lig).split("_")) and (rec_gene in str(rec).split("_"))

        for lig_gene, rec_gene in FOCUS_PAIRS:
            hit = df[df.apply(lambda r: contains_pair(r[ligand_col], r[receptor_col], lig_gene, rec_gene), axis=1)].copy()
            if len(hit) == 0:
                continue
            hit["label"] = label
            hit["focus_pair"] = f"{lig_gene}->{rec_gene}"
            focus_rows.append(hit)

        # MHC-II related (ligand contains HLA-D*)
        mhc = df[df[ligand_col].astype(str).str.startswith(MHC_PREFIXES)].copy()
        if len(mhc) > 0:
            mhc["label"] = label
            mhc["focus_pair"] = "MHC-II (HLA-D*)"
            focus_rows.append(mhc)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_tables, "liana_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)

    if focus_rows:
        focus_df = pd.concat(focus_rows, ignore_index=True)
        focus_path = os.path.join(out_tables, "liana_focus_pairs.tsv")
        focus_df.to_csv(focus_path, sep="\t", index=False)
    else:
        focus_df = pd.DataFrame()
        focus_path = os.path.join(out_tables, "liana_focus_pairs.tsv")
        focus_df.to_csv(focus_path, sep="\t", index=False)

    print("Wrote:", summary_path)
    print("Wrote:", focus_path)

    # Simple bar plot: n_rows per dataset
    plt.figure()
    plt.bar(summary_df["label"], summary_df["n_rows"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("LIANA rows")
    plt.title("LIANA result size per dataset")
    plt.tight_layout()
    fig_path = os.path.join(out_figs, "liana_rows_per_dataset.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Wrote:", fig_path)

if __name__ == "__main__":
    main()

