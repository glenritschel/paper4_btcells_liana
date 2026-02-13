#!/usr/bin/env python
"""
04_score_programs.py

Score signature/program gene sets.
- Recommended: pass configs/run.yaml and define scoring.scores in configs/scoring.yaml
- Backward-compatible: pass configs/scoring.yaml with top-level "scores:".

Usage:
  python scripts/04_score_programs.py --config configs/run.yaml --indir work/annot --outdir work/scored
"""

import argparse
import os
from pathlib import Path
import yaml
import scanpy as sc


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config_with_includes(config_path: str) -> dict:
    """
    Supports:
      include:
        - datasets.yaml
        - qc.yaml
        - scoring.yaml
        - liana.yaml
    Include paths are resolved relative to the directory containing config_path.
    Later files override earlier ones; the root file overrides all includes.
    """
    root = Path(config_path).resolve()
    if not root.exists():
        raise SystemExit(f"[ERROR] Config file not found: {root}")

    base_dir = root.parent
    root_cfg = _load_yaml(root)

    merged: dict = {}
    for inc in root_cfg.get("include", []) or []:
        inc_path = (base_dir / inc).resolve()
        if not inc_path.exists():
            raise SystemExit(f"[ERROR] Config file not found: {inc_path}")
        merged.update(_load_yaml(inc_path))

    merged.update({k: v for k, v in root_cfg.items() if k != "include"})
    return merged


def present(var_names, genes):
    varset = set(map(str, var_names))
    return [g for g in genes if g in varset]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_config_with_includes(args.config)

    # Preferred: scoring: { scores: {...} }
    scoring_cfg = cfg.get("scoring", {}) if isinstance(cfg.get("scoring", {}), dict) else {}
    scores = scoring_cfg.get("scores")

    # Back-compat: top-level scores: {...}
    if not scores:
        scores = cfg.get("scores")

    if not scores or not isinstance(scores, dict):
        raise SystemExit(
            "No scores found. Expected either:\n"
            "  scoring:\n"
            "    scores: {...}\n"
            "or legacy:\n"
            "  scores: {...}"
        )

    os.makedirs(args.outdir, exist_ok=True)
    files = [f for f in os.listdir(args.indir) if f.endswith(".annot.h5ad")]
    if not files:
        raise SystemExit(f"No *.annot.h5ad in {args.indir}")

    for fn in sorted(files):
        label = fn.replace(".annot.h5ad", "")
        path = os.path.join(args.indir, fn)
        print(f"\n=== Score programs: {label} ===")
        adata = sc.read_h5ad(path)

        # Ensure sample_id exists (helps downstream)
        if "sample_id" not in adata.obs.columns:
            adata.obs["sample_id"] = label

        for score_name, genes in scores.items():
            if not isinstance(genes, (list, tuple)):
                print(f"WARNING: score {score_name} is not a list; skipping")
                continue
            genes2 = present(adata.var_names, genes)
            if len(genes2) == 0:
                print(f"WARNING: score {score_name} has 0 genes present; writing zeros")
                adata.obs[score_name] = 0.0
                continue
            sc.tl.score_genes(adata, gene_list=genes2, score_name=score_name, use_raw=False)
            print(f"Scored {score_name}: {len(genes2)}/{len(genes)} genes present")

        out = os.path.join(args.outdir, f"{label}.scored.h5ad")
        adata.write(out)
        print("Wrote:", out)


if __name__ == "__main__":
    main()

