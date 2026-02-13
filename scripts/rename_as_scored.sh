#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <file1.h5ad> [file2.h5ad ...]"
  echo "Creates sibling *.scored.h5ad copies."
  exit 2
fi

for f in "$@"; do
  if [[ "$f" != *.h5ad ]]; then
    echo "[SKIP] $f (not .h5ad)"
    continue
  fi
  out="${f%.h5ad}.scored.h5ad"
  cp -v "$f" "$out"
done

