#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd
import scanpy as sc

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .h5ad")
    ap.add_argument("--groupby", default="cell_type")
    ap.add_argument("--sample-key", default="sample_id")
    ap.add_argument("--show-suffix", action="store_true", help="Report obs_names suffix diversity (barcode suffix)")
    args = ap.parse_args()

    ad = sc.read_h5ad(args.input)

    print(f"[INPUT] {args.input}")
    print(f"[SHAPE] cells={ad.n_obs} genes={ad.n_vars}")
    print(f"[OBS] columns={len(ad.obs.columns)}")

    if args.sample_key in ad.obs:
        print(f"[SAMPLES] {args.sample_key} nunique={ad.obs[args.sample_key].nunique()}")
        print(ad.obs[args.sample_key].value_counts().head(10).to_string())
    else:
        print(f"[SAMPLES] {args.sample_key} not present")

    if args.groupby in ad.obs:
        print(f"\n[GROUPBY] {args.groupby} counts:")
        print(ad.obs[args.groupby].value_counts().head(30).to_string())
    else:
        print(f"\n[GROUPBY] {args.groupby} not present")

    if args.show_suffix:
        suffix = pd.Index(ad.obs_names).str.rsplit("-", n=3).str[-3:].str.join("-")
        print(f"\n[OBS_NAMES SUFFIX] nunique={suffix.nunique()}")
        print(suffix.value_counts().head(10).to_string())

if __name__ == "__main__":
    main()

