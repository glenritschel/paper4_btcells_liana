#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <overlap_by_delta.tsv> [--head N]"
  exit 2
fi

FILE="$1"
N=25
if [[ $# -ge 3 && "$2" == "--head" ]]; then
  N="$3"
fi

echo "# Up in Bhigh (delta_rank most negative)"
awk 'NR==1 || $8 < 0' "$FILE" | head -n "$N"

echo
echo "# Up in Blow (delta_rank most positive)"
awk 'NR==1 || $8 > 0' "$FILE" | head -n "$N"

