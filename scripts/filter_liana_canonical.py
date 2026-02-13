#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="*.liana_top_bt.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resource", default="consensus", help="LIANA resource name (e.g., consensus, omnipath)")
    ap.add_argument("--min-cols", action="store_true", help="only require source/target/ligand_complex/receptor_complex")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t")

    # Build lr_id if needed
    if "lr_id" not in df.columns:
        req = {"source","target","ligand_complex","receptor_complex"}
        if not req.issubset(df.columns):
            raise SystemExit("Input lacks lr_id and required columns to build it.")
        df["lr_id"] = df["source"].astype(str) + "|" + df["target"].astype(str) + "|" + df["ligand_complex"].astype(str) + "|" + df["receptor_complex"].astype(str)

    # Load resource and attempt canonical filtering based on available annotations
    import liana as li
    from liana.resource import select_resource
    res = select_resource(args.resource).copy()

    # Normalize column names we might see
    cols = set(res.columns)

    # Construct a boolean mask "canonical" if possible
    canonical = None

    # Option 1: explicit boolean flags
    flag_sets = [
        ("ligand_is_secreted", "receptor_is_membrane"),
        ("ligand_is_membrane", "receptor_is_membrane"),
        ("ligand_is_surface", "receptor_is_surface"),
    ]
    for a,b in flag_sets:
        if a in cols and b in cols:
            canonical = res[a].astype(bool) & res[b].astype(bool)
            break

    # Option 2: location strings
    if canonical is None:
        loc_pairs = [("ligand_location","receptor_location")]
        for a,b in loc_pairs:
            if a in cols and b in cols:
                la = res[a].astype(str).str.lower()
                rb = res[b].astype(str).str.lower()
                canonical = (
                    (la.str.contains("secret") | la.str.contains("membrane") | la.str.contains("surface") | la.str.contains("extracellular")) &
                    (rb.str.contains("membrane") | rb.str.contains("surface") | rb.str.contains("plasma"))
                )
                break

    if canonical is None:
        raise SystemExit(
            f"Resource '{args.resource}' lacks recognizable localization annotations. "
            "Either pick another resource or use a denylist approach."
        )

    res_f = res.loc[canonical].copy()

    # Resource typically uses ligand/receptor columns (may vary). Try common names.
    ligand_col = "ligand" if "ligand" in res_f.columns else ("ligand_complex" if "ligand_complex" in res_f.columns else None)
    receptor_col = "receptor" if "receptor" in res_f.columns else ("receptor_complex" if "receptor_complex" in res_f.columns else None)
    if ligand_col is None or receptor_col is None:
        raise SystemExit("Cannot identify ligand/receptor columns in resource table.")

    keep = set(zip(res_f[ligand_col].astype(str), res_f[receptor_col].astype(str)))

    before = len(df)
    df2 = df[df.apply(lambda r: (str(r["ligand_complex"]), str(r["receptor_complex"])) in keep, axis=1)].copy()
    after = len(df2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(out, sep="\t", index=False)

    print(f"Filtered {before} -> {after} rows using resource '{args.resource}' canonical annotations.")
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()

