#!/usr/bin/env python3

import sys
import yaml
from pathlib import Path


# -------------------------
# YAML Loading + Merging
# -------------------------

def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"[ERROR] Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    cfg = load_yaml(path)
    includes = cfg.pop("include", [])
    base = {}

    for inc in includes:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = path.parent / inc
        base = deep_merge(base, load_yaml(inc_path))

    return deep_merge(base, cfg)


# -------------------------
# Validation
# -------------------------

def validate_config(cfg: dict):
    required_top = ["datasets", "qc"]
    for key in required_top:
        if key not in cfg:
            raise SystemExit(f"[ERROR] Missing required top-level key: '{key}'")

    if not isinstance(cfg["datasets"], list) or len(cfg["datasets"]) == 0:
        raise SystemExit("[ERROR] 'datasets' must be a non-empty list")

    # Validate each dataset entry
    for i, ds in enumerate(cfg["datasets"]):
        if not isinstance(ds, dict):
            raise SystemExit(f"[ERROR] Dataset entry {i} is not a dict")

        if "geo" not in ds:
            raise SystemExit(f"[ERROR] Dataset entry {i} missing 'geo' key")

        if "label" not in ds:
            raise SystemExit(f"[ERROR] Dataset entry {i} missing 'label' key")

    # Validate QC keys
    qc_required = ["min_genes", "max_genes", "max_mt_pct"]
    for key in qc_required:
        if key not in cfg["qc"]:
            raise SystemExit(f"[ERROR] Missing required qc key: '{key}'")

    print("[OK] Config validation passed.")


# -------------------------
# Main
# -------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_config.py configs/run.yaml")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    cfg = load_config(config_path)

    validate_config(cfg)

    print("\n--- Merged Config ---")
    print(yaml.dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()

