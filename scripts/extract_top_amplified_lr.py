#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="top_lr")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")

    # Sort by strongest promotion in Bhigh (most negative delta_rank)
    df_sorted = df.sort_values("delta_rank")

    # Top interactions promoted in Bhigh
    promoted = df_sorted.head(args.top)

    # Strongest ligand amplification
    amplified = df.sort_values("lig_log2fc", ascending=False).head(args.top)

    # Strongest pair amplification
    pair_amp = df.sort_values("pair_amp_log2fc", ascending=False).head(args.top)

    promoted.to_csv(outdir / f"{args.prefix}.promoted.tsv", sep="\t", index=False)
    amplified.to_csv(outdir / f"{args.prefix}.ligand_amp.tsv", sep="\t", index=False)
    pair_amp.to_csv(outdir / f"{args.prefix}.pair_amp.tsv", sep="\t", index=False)

    print("Wrote top LR interaction tables.")

if __name__ == "__main__":
    main()

