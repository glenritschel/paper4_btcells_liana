#!/usr/bin/env python3
"""
Compare LIANA results across two datasets.

Typical usage (top lists, universe from full files inferred):
  python scripts/07_compare_liana_datasets.py \
    --a results/liana/A.BT.liana_top_bt.tsv \
    --b results/liana/B.BT.liana_top_bt.tsv \
    --outdir results/compare \
    --topk 0 \
    --label-a A \
    --label-b B

You can also explicitly provide full/universe files:
  --a-full results/liana/A.BT.liana.tsv
  --b-full results/liana/B.BT.liana.tsv

Outputs:
  - overlap TSV with ranks/metrics per LR pair
  - summary JSON with stats, hypergeom p-values, permutation p-values, correlations
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import hypergeom, spearmanr, pearsonr
except Exception as e:
    raise SystemExit(f"[ERROR] scipy is required: {e}")


# -----------------------------
# Helpers
# -----------------------------
REQUIRED_COLS = ["source", "target", "ligand_complex", "receptor_complex"]


def _infer_full_path(path: str) -> Optional[str]:
    """
    If given a top file like *.liana_top_bt.tsv, try to infer the full file:
      *.liana.tsv
    """
    base = path
    for suffix in [".liana_top_bt.tsv", ".liana_top_tb.tsv", ".liana_top.tsv"]:
        if base.endswith(suffix):
            cand = base[: -len(suffix)] + ".liana.tsv"
            return cand if os.path.exists(cand) else None
    # If it's already full
    if base.endswith(".liana.tsv") and os.path.exists(base):
        return base
    return None


def lr_id_from_row(row: pd.Series) -> str:
    return f"{row['source']}|{row['target']}|{row['ligand_complex']}|{row['receptor_complex']}"


def ensure_cols(df: pd.DataFrame, path: str) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path}: missing required columns: {missing}. Have: {list(df.columns)}")


def compute_combined_rank(df: pd.DataFrame) -> pd.Series:
    """
    Prefer LIANA-provided ranks if present; otherwise synthesize.

    We want a "smaller is better" scalar per LR. Most LIANA outputs include
    'specificity_rank' and 'magnitude_rank' (both 0..1-ish).
    """
    if "aggregate_rank" in df.columns:
        # Some LIANA versions output this
        return pd.to_numeric(df["aggregate_rank"], errors="coerce")

    if "specificity_rank" in df.columns and "magnitude_rank" in df.columns:
        a = pd.to_numeric(df["specificity_rank"], errors="coerce")
        b = pd.to_numeric(df["magnitude_rank"], errors="coerce")
        return (a + b) / 2.0

    # Fall back to something rankable:
    # If lrscore exists: larger is better; convert to rank-like by negative
    if "lrscore" in df.columns:
        x = pd.to_numeric(df["lrscore"], errors="coerce")
        # convert to ascending "rank score"
        return -x

    # Else: scaled_weight or expr_prod etc
    for col in ["scaled_weight", "expr_prod", "lr_means", "lr_logfc"]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            return -x

    # Worst case: constant
    return pd.Series(np.nan, index=df.index)


def load_liana(path: str, topk: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load LIANA TSV, compute LR id and combined_rank.
    Return:
      - df_raw: all rows (possibly truncated if topk>0)
      - df_best: unique LR id rows (best per LR id by combined_rank ascending)
    """
    df = pd.read_csv(path, sep="\t")
    ensure_cols(df, path)

    df = df.copy()
    df["lr_id"] = df.apply(lr_id_from_row, axis=1)
    df["combined_rank"] = compute_combined_rank(df)

    # If topk requested: take best topk by combined_rank (ascending), after dedup best-per-lr
    # BUT first: collapse duplicates consistently.
    df_best = (
        df.sort_values(["combined_rank"], ascending=True, na_position="last")
        .groupby("lr_id", as_index=False)
        .first()
    )

    if topk and topk > 0:
        df_best = df_best.sort_values("combined_rank", ascending=True, na_position="last").head(topk)

    # make a "rank index" for correlation: 1..K
    df_best = df_best.reset_index(drop=True)
    df_best["rank"] = np.arange(1, len(df_best) + 1, dtype=np.int32)

    return df, df_best


def _two_sided_from_tails(p_upper: float, p_lower: float) -> float:
    p = 2.0 * min(p_upper, p_lower)
    return float(min(1.0, max(0.0, p)))


def perm_test_overlap(N: int, K: int, n: int, k_obs: int, reps: int, seed: int) -> Dict[str, float]:
    """
    Permutation test for overlap size between two random subsets of sizes K and n from universe N.
    """
    rng = np.random.default_rng(seed)
    # Efficient simulation: sample A as indices, then sample B and count intersection
    # We'll do this in a loop to keep memory stable.
    overlaps = np.empty(reps, dtype=np.int32)

    universe = np.arange(N, dtype=np.int32)
    for i in range(reps):
        a = rng.choice(universe, size=K, replace=False)
        b = rng.choice(universe, size=n, replace=False)
        overlaps[i] = np.intersect1d(a, b, assume_unique=False).size

    mu = float(np.mean(overlaps))
    sd = float(np.std(overlaps, ddof=1)) if reps > 1 else float("nan")
    z = (k_obs - mu) / sd if sd and sd > 0 else float("nan")

    p_upper = float((overlaps >= k_obs).mean())
    p_lower = float((overlaps <= k_obs).mean())
    p_two = _two_sided_from_tails(p_upper, p_lower)

    return {
        "perm_reps": float(reps),
        "perm_mean": mu,
        "perm_sd": sd,
        "perm_z": z,
        "perm_p_upper": p_upper,
        "perm_p_lower": p_lower,
        "perm_p_two_sided": p_two,
    }


@dataclass
class CompareResult:
    overlap_df: pd.DataFrame
    summary: Dict


def compare(
    a_top_path: str,
    b_top_path: str,
    outdir: str,
    topk: int,
    label_a: str,
    label_b: str,
    a_full_path: Optional[str],
    b_full_path: Optional[str],
    universe_mode: str,
    perm_reps: int,
    seed: int,
) -> CompareResult:
    os.makedirs(outdir, exist_ok=True)

    # Load TOP lists (or full if user passed full)
    _, a_top = load_liana(a_top_path, topk=topk)
    _, b_top = load_liana(b_top_path, topk=topk)

    a_top = a_top[(a_top["source"]=="B_cell") & (a_top["target"]=="T_cell")]
    b_top = b_top[(b_top["source"]=="B_cell") & (b_top["target"]=="T_cell")]

    # Determine universe
    if universe_mode not in ("full_union", "top_union"):
        raise RuntimeError(f"--universe must be full_union or top_union (got {universe_mode})")

    if universe_mode == "top_union":
        universe_ids = sorted(set(a_top["lr_id"]).union(set(b_top["lr_id"])))
        a_full_used = None
        b_full_used = None
    else:
        # full_union
        if not a_full_path:
            a_full_path = _infer_full_path(a_top_path)
        if not b_full_path:
            b_full_path = _infer_full_path(b_top_path)

        if not a_full_path or not os.path.exists(a_full_path):
            raise RuntimeError(
                f"Universe mode full_union requires --a-full or an inferable full file. "
                f"Could not find full file for: {a_top_path}"
            )
        if not b_full_path or not os.path.exists(b_full_path):
            raise RuntimeError(
                f"Universe mode full_union requires --b-full or an inferable full file. "
                f"Could not find full file for: {b_top_path}"
            )

        a_full_used = a_full_path
        b_full_used = b_full_path

        _, a_full_best = load_liana(a_full_path, topk=0)
        _, b_full_best = load_liana(b_full_path, topk=0)

        a_full_best = a_full_best[(a_full_best["source"]=="B_cell") & (a_full_best["target"]=="T_cell")]
        b_full_best = b_full_best[(b_full_best["source"]=="B_cell") & (b_full_best["target"]=="T_cell")]


        universe_ids = sorted(set(a_full_best["lr_id"]).union(set(b_full_best["lr_id"])))

    # Sets and counts
    set_a = set(a_top["lr_id"])
    set_b = set(b_top["lr_id"])
    overlap = sorted(set_a.intersection(set_b))

    K = len(set_a)
    n = len(set_b)
    k_obs = len(overlap)
    N = len(universe_ids)

    # Jaccard based on TOP sets (not on universe)
    jacc = float(k_obs / len(set_a.union(set_b))) if (set_a or set_b) else float("nan")

    # Hypergeometric stats
    # X ~ Hypergeom(N, K, n) = overlap between set A (size K) and set B (size n)
    # Upper-tail enrichment: P(X >= k_obs) = sf(k_obs-1)
    # Lower-tail depletion: P(X <= k_obs) = cdf(k_obs)
    hg = hypergeom(M=N, n=K, N=n)  # scipy uses M=population, n=successes, N=draws
    p_upper = float(hg.sf(k_obs - 1))
    p_lower = float(hg.cdf(k_obs))
    p_two = _two_sided_from_tails(p_upper, p_lower)

    expected = float(K * n / N) if N > 0 else float("nan")
    fold = float(k_obs / expected) if expected and expected > 0 else float("nan")

    # Rank correlations among shared LR ids
    # Use "rank" column from a_top / b_top (1..K).
    a_lookup = a_top.set_index("lr_id")
    b_lookup = b_top.set_index("lr_id")

    shared_ranks_a = []
    shared_ranks_b = []
    for lr in overlap:
        # ensure scalar
        ra = a_lookup.loc[lr, "rank"]
        rb = b_lookup.loc[lr, "rank"]
        # If duplicates slipped through, take min
        if isinstance(ra, pd.Series):
            ra = ra.min()
        if isinstance(rb, pd.Series):
            rb = rb.min()
        shared_ranks_a.append(float(ra))
        shared_ranks_b.append(float(rb))

    if len(overlap) >= 3:
        sp = spearmanr(shared_ranks_a, shared_ranks_b)
        pe = pearsonr(shared_ranks_a, shared_ranks_b)
        spearman_rho = float(sp.statistic)
        spearman_p = float(sp.pvalue)
        pearson_r = float(pe.statistic)
        pearson_p = float(pe.pvalue)
    else:
        spearman_rho = float("nan")
        spearman_p = float("nan")
        pearson_r = float("nan")
        pearson_p = float("nan")

    # Permutation test (optional)
    perm_stats = {}
    if perm_reps and perm_reps > 0 and N > 0:
        perm_stats = perm_test_overlap(N=N, K=K, n=n, k_obs=k_obs, reps=perm_reps, seed=seed)

    # Build overlap table
    rows = []
    for lr in overlap:
        ra = a_lookup.loc[lr, "rank"]
        rb = b_lookup.loc[lr, "rank"]
        if isinstance(ra, pd.Series):
            ra = ra.min()
        if isinstance(rb, pd.Series):
            rb = rb.min()

        # Pull human-readable parts from A (should match B)
        r0 = a_lookup.loc[lr]
        if isinstance(r0, pd.DataFrame):
            r0 = r0.iloc[0]

        rows.append(
            {
                "lr_id": lr,
                "source": r0["source"],
                "target": r0["target"],
                "ligand_complex": r0["ligand_complex"],
                "receptor_complex": r0["receptor_complex"],
                "rank_a": float(ra),
                "rank_b": float(rb),
            }
        )


    overlap_df = pd.DataFrame(rows)

    # Filenames
    top_tag = f"top{topk}" if topk and topk > 0 else "topALL"
    out_prefix = os.path.join(outdir, f"compare_{label_a}_vs_{label_b}_{top_tag}")

    overlap_path = out_prefix + ".overlap.tsv"
    overlap_delta_path = out_prefix + ".overlap_by_delta.tsv"
    summary_path = out_prefix + ".summary.json"

    if overlap_df.empty:
        overlap_df.to_csv(overlap_path, sep="\t", index=False)
        overlap_df.to_csv(overlap_delta_path, sep="\t", index=False)
        print(f"[OK] Wrote overlap (empty): {overlap_path}")
        print(f"[OK] Wrote overlap_by_delta (empty): {overlap_delta_path}")
    else:
        # ---- Canonical table (NO delta columns) ----
        overlap_df_canon = overlap_df.sort_values(
            ["rank_a", "rank_b"],
            ascending=[True, True],
        )
        overlap_df_canon.to_csv(overlap_path, sep="\t", index=False)
        print(f"[OK] Wrote overlap: {overlap_path}")

        # ---- Delta table (with rank-shift metrics) ----
        overlap_df_delta = overlap_df.copy()
        overlap_df_delta["delta_rank"] = (
            overlap_df_delta["rank_b"] - overlap_df_delta["rank_a"]
        )
        overlap_df_delta["abs_delta_rank"] = overlap_df_delta["delta_rank"].abs()

        overlap_df_delta = overlap_df_delta.sort_values(
            ["abs_delta_rank", "rank_a", "rank_b"],
            ascending=[False, True, True],
        )

        overlap_df_delta.to_csv(overlap_delta_path, sep="\t", index=False)
        print(f"[OK] Wrote overlap_by_delta: {overlap_delta_path}")


    summary = {
        "labels": {"a": label_a, "b": label_b},
        "inputs": {
            "a_top": a_top_path,
            "b_top": b_top_path,
            "a_full": a_full_used,
            "b_full": b_full_used,
            "universe_mode": universe_mode,
        },
        "counts": {
            "a_topk": K,
            "b_topk": n,
            "overlap": k_obs,
            "universe_N": N,
            "top_union_size": len(set_a.union(set_b)),
        },
        "overlap_metrics": {
            "jaccard_top_sets": jacc,
            "expected_overlap_under_random": expected,
            "fold_enrichment_vs_random": fold,
        },
        "hypergeom": {
            "p_upper_enrichment": p_upper,
            "p_lower_depletion": p_lower,
            "p_two_sided": p_two,
        },
        "rank_correlation_shared": {
            "n_shared": k_obs,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
        },
        "permutation": perm_stats,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote summary: {summary_path}")
    print(
        f"[SUMMARY] overlap={k_obs} jaccard={jacc:.4f} "
        f"expected={expected:.2f} fold={fold:.3f} "
        f"hypergeom_p_upper={p_upper:.3e} spearman_rho={spearman_rho:.3f} (n_shared={k_obs})"
    )

    return CompareResult(overlap_df=overlap_df, summary=summary)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="LIANA TSV for dataset A (typically *_top_bt.tsv or *_top_tb.tsv)")
    ap.add_argument("--b", required=True, help="LIANA TSV for dataset B (typically *_top_bt.tsv or *_top_tb.tsv)")
    ap.add_argument("--a-full", default=None, help="Full LIANA TSV for dataset A (universe); optional if inferable")
    ap.add_argument("--b-full", default=None, help="Full LIANA TSV for dataset B (universe); optional if inferable")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--topk", type=int, default=0, help="If >0, compare only top K interactions from each file")
    ap.add_argument("--label-a", default="A", help="Label for dataset A")
    ap.add_argument("--label-b", default="B", help="Label for dataset B")
    ap.add_argument(
        "--universe",
        choices=["full_union", "top_union"],
        default="full_union",
        help="Universe definition for hypergeometric/permutation tests (default: full_union)",
    )
    ap.add_argument("--perm-reps", type=int, default=10000, help="Permutation reps (0 disables)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for permutation test")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    compare(
        a_top_path=args.a,
        b_top_path=args.b,
        outdir=args.outdir,
        topk=args.topk,
        label_a=args.label_a,
        label_b=args.label_b,
        a_full_path=args.a_full,
        b_full_path=args.b_full,
        universe_mode=args.universe,
        perm_reps=args.perm_reps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

