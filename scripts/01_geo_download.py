#!/usr/bin/env python
"""
01_geo_download.py

Download GEO Series metadata + supplementary files into a reproducible work directory.

Usage:
  python scripts/01_geo_download.py --config configs/datasets.yaml --outdir work/geo
"""

import argparse
import os
import sys
import yaml
import GEOparse

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/datasets.yaml")
    ap.add_argument("--outdir", required=True, help="output directory, e.g. work/geo")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    datasets = cfg.get("datasets", [])
    if not datasets:
        raise SystemExit("No datasets found in configs/datasets.yaml under key: datasets")

    os.makedirs(args.outdir, exist_ok=True)

    for ds in datasets:
        geo = ds["geo"]
        label = ds.get("label", geo)
        ds_dir = os.path.join(args.outdir, label)
        os.makedirs(ds_dir, exist_ok=True)

        print(f"\n=== Downloading GEO: {geo} ({label}) ===")
        gse = GEOparse.get_GEO(geo=geo, destdir=ds_dir)

        # Save a quick sample metadata table for inspection
        rows = []
        for gsm_name, gsm in gse.gsms.items():
            meta = gsm.metadata
            rows.append({
                "gsm": gsm_name,
                "title": meta.get("title", [""])[0],
                "source_name_ch1": meta.get("source_name_ch1", [""])[0],
                "organism_ch1": meta.get("organism_ch1", [""])[0],
                "characteristics_ch1": "; ".join(meta.get("characteristics_ch1", [])),
                "supplementary_file": "; ".join(meta.get("supplementary_file", [])),
            })

        import pandas as pd
        md_path = os.path.join(ds_dir, f"{label}_gsm_metadata.tsv")
        pd.DataFrame(rows).to_csv(md_path, sep="\t", index=False)
        print(f"Wrote sample metadata: {md_path}")

        print("Downloading supplementary files (if any)...")
        try:
            gse.download_supplementary_files(ds_dir)
        except Exception as e:
            print(f"WARNING: supplementary download error for {geo}: {e}", file=sys.stderr)

        print(f"Done: {geo} -> {ds_dir}")

if __name__ == "__main__":
    main()

