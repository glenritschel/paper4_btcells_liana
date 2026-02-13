#!/usr/bin/env python
"""
05_run_liana.py

Run LIANA receptor-ligand inference on B<->T interactions.

Robust per-sample mode:
- Filters out samples that don't have enough cells in BOTH sender and receiver.
- Falls back to pooled rank_aggregate if too few valid samples remain.
"""

import argparse
import os
import yaml
import pandas as pd
import scanpy as sc
import liana as li
import numpy as np
import scipy.sparse as sp

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _valid_samples_bt(
    adata: sc.AnnData,
    sample_key: str,
    groupby: str,
    sender: str,
    receiver: str,
    min_cells: int,
) -> list[str]:
    """
    Return sample ids where BOTH sender and receiver have >= min_cells.
    """
    df = adata.obs[[sample_key, groupby]].copy()

    # counts per (sample, group)
    ct = (
        df.groupby([sample_key, groupby], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    # ensure columns exist
    if sender not in ct.columns:
        ct[sender] = 0
    if receiver not in ct.columns:
        ct[receiver] = 0

    ok = ct[(ct[sender] >= min_cells) & (ct[receiver] >= min_cells)]
    return ok.index.astype(str).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config).get("liana", {})
    groupby = cfg.get("groupby", "cell_type")
    sample_key = cfg.get("sample_key", "sample_id")
    use_raw = bool(cfg.get("use_raw", False))
    min_cells = int(cfg.get("min_cells_per_group", 50))
    sender = cfg.get("sender", "B_cell")
    receiver = cfg.get("receiver", "T_cell")

    os.makedirs(args.outdir, exist_ok=True)

    files = [f for f in os.listdir(args.indir) if f.endswith(".scored.h5ad")]
    if not files:
        raise SystemExit(f"No *.scored.h5ad in {args.indir}")

    for fn in sorted(files):
        label = fn.replace(".scored.h5ad", "")
        path = os.path.join(args.indir, fn)
        print(f"\n=== LIANA: {label} ===")
        adata = sc.read_h5ad(path)

        if groupby not in adata.obs.columns:
            raise RuntimeError(
                f"{label}: '{groupby}' not in adata.obs columns: {list(adata.obs.columns)}"
            )

        # Subset to B and T only (speed/clarity)
        adata_bt = adata[adata.obs[groupby].isin([sender, receiver])].copy()


        def _x_max(X) -> float:
            # Works for dense and sparse without densifying
            if sp.issparse(X):
                # scipy sparse max() returns a 1x1 matrix or scalar depending on version
                m = X.max()
                try:
                    return float(m)
                except TypeError:
                    return float(m.toarray().item())
            return float(np.max(X))

        # Preserve original counts once (if not already present)
        if "counts" not in adata_bt.layers:
            adata_bt.layers["counts"] = adata_bt.X.copy()

        # Prefer Scanpy's own marker for log1p preprocessing
        already_log1p = isinstance(adata_bt.uns.get("log1p", None), dict)

        # Fallback heuristic (only used if log1p marker absent)
        # Typical log1p normalized values are usually < ~15-20
        x_max = _x_max(adata_bt.X)
        looks_like_counts = (not already_log1p) and (x_max > 20)

        if looks_like_counts:
            print(f"[LIANA] Detected count-like .X (max={x_max:.2f}). Normalizing + log1p.")
            sc.pp.normalize_total(adata_bt, target_sum=1e4)
            sc.pp.log1p(adata_bt)
        else:
            print(f"[LIANA] .X looks log1p-normalized (max={x_max:.2f}, log1p_marker={already_log1p}). Proceeding.")

        # Dataset-level gate
        counts = adata_bt.obs[groupby].value_counts()
        print("Group counts:\n", counts)
        if counts.get(sender, 0) < min_cells or counts.get(receiver, 0) < min_cells:
            print(f"[WARN] insufficient cells for LIANA (min {min_cells}). Skipping {label}.")
            continue

        # Decide per-sample mode
        has_sample = (
            sample_key in adata_bt.obs.columns
            and adata_bt.obs[sample_key].nunique() > 1
        )

        # If per-sample, filter to valid samples first
        if has_sample:
            valid = _valid_samples_bt(
                adata_bt, sample_key, groupby, sender, receiver, min_cells
            )
            print(f"Per-sample mode: {adata_bt.obs[sample_key].nunique()} total samples")
            print(f"Keeping {len(valid)} samples with >= {min_cells} {sender} AND >= {min_cells} {receiver}")

            if len(valid) == 0:
                print("[WARN] No samples meet per-sample thresholds. Falling back to pooled mode.")
                has_sample = False
            else:
                adata_bt = adata_bt[adata_bt.obs[sample_key].astype(str).isin(valid)].copy()

                # After filtering, if only 1 sample remains, pooled is more sensible
                if adata_bt.obs[sample_key].nunique() <= 1:
                    print("[WARN] <=1 valid sample after filtering. Falling back to pooled mode.")
                    has_sample = False

        # Defensive: never call LIANA with 0 cells
        if adata_bt.n_obs == 0:
            print("[WARN] 0 cells after filtering. Skipping.")
            continue


        # --- Remove genes expressed in 0 cells (stabilizes LIANA) ---
        n_vars_before = adata_bt.n_vars
        sc.pp.filter_genes(adata_bt, min_cells=1)
        n_vars_after = adata_bt.n_vars
        if n_vars_after != n_vars_before:
            print(f"[LIANA] Removed {n_vars_before - n_vars_after} all-zero genes.")

        # -------------------------
        # Run LIANA
        # -------------------------
        try:
            if has_sample:
                print(f"Running rank_aggregate.by_sample(groupby={groupby}, sample_key={sample_key})")
                li.mt.rank_aggregate.by_sample(
                    adata_bt,
                    groupby=groupby,
                    sample_key=sample_key,
                    use_raw=use_raw,
                    verbose=True,
                )
            else:
                print(f"Running rank_aggregate(groupby={groupby})")
                li.mt.rank_aggregate(
                    adata_bt,
                    groupby=groupby,
                    use_raw=use_raw,
                    verbose=True,
                )
        except Exception as e:
            raise RuntimeError(f"LIANA failed for {label}: {e}")

        res = adata_bt.uns.get("liana_res")
        if res is None:
            raise RuntimeError(f"{label}: No adata.uns['liana_res'] produced")

        out_all = os.path.join(args.outdir, f"{label}.liana.tsv")
        res.to_csv(out_all, sep="\t", index=False)
        print("Wrote:", out_all)

        def top_direction(df, src, tgt, n=200):
            df2 = df[(df["source"] == src) & (df["target"] == tgt)].copy()
            if "aggregate_rank" in df2.columns:
                df2 = df2.sort_values("aggregate_rank", ascending=True)
            return df2.head(n)

        bt = top_direction(res, sender, receiver, n=200)
        tb = top_direction(res, receiver, sender, n=200)

        out_bt = os.path.join(args.outdir, f"{label}.liana_top_bt.tsv")
        out_tb = os.path.join(args.outdir, f"{label}.liana_top_tb.tsv")
        bt.to_csv(out_bt, sep="\t", index=False)
        tb.to_csv(out_tb, sep="\t", index=False)
        print("Wrote:", out_bt)
        print("Wrote:", out_tb)


if __name__ == "__main__":
    main()

