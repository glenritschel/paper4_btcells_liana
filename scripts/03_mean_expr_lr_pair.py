#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import scanpy as sc


def mean_expr(ad: sc.AnnData, gene: str) -> float:
    if gene not in ad.var_names:
        return float("nan")
    x = ad[:, [gene]].X
    # robust to sparse + views
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x).ravel()
    return float(np.mean(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-a", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--input-b", required=True)
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--cell-type-col", default="cell_type")
    ap.add_argument("--b-label", default="B_cell")
    ap.add_argument("--t-label", default="T_cell")
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--receptor", required=True)
    args = ap.parse_args()

    for path, lab in [(args.input_a, args.label_a), (args.input_b, args.label_b)]:
        ad = sc.read_h5ad(path)
        b = ad[ad.obs[args.cell_type_col] == args.b_label]
        t = ad[ad.obs[args.cell_type_col] == args.t_label]
        print(
            f"{lab}  B mean {args.ligand} {mean_expr(b, args.ligand):.6f}  | "
            f"T mean {args.receptor} {mean_expr(t, args.receptor):.6f}"
        )


if __name__ == "__main__":
    main()

