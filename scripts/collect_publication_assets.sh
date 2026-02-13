#!/usr/bin/env bash
set -euo pipefail

#############################################
# Configuration
#############################################

ROOT_DIR="$(git rev-parse --show-toplevel)"
RESULTS_DIR="${ROOT_DIR}/results"
PAPER_DIR="${ROOT_DIR}/paper"

FIG_DIR="${PAPER_DIR}/figures"
TABLE_DIR="${PAPER_DIR}/tables"
ARTIFACT_DIR="${PAPER_DIR}/artifacts"

INCLUDE_ARTIFACTS=true   # set false if you only want clean submission assets

#############################################
# Prepare directories
#############################################

echo "Preparing paper directories..."

mkdir -p "$FIG_DIR"
mkdir -p "$TABLE_DIR"
mkdir -p "$ARTIFACT_DIR"

#############################################
# 1. Collect Figures
#############################################

echo "Collecting figures..."

find "$RESULTS_DIR" \
  -type f \
  \( -iname "*.png" -o -iname "*.pdf" -o -iname "*.svg" \) \
  ! -iname "*debug*" \
  ! -iname "*tmp*" \
  -print0 | while IFS= read -r -d '' file; do
    cp -f "$file" "$FIG_DIR/"
done

#############################################
# 2. Collect Tables (publication ready)
#############################################

echo "Collecting tables..."

find "$RESULTS_DIR" \
  -type f \
  \( -iname "*.tsv" -o -iname "*.csv" \) \
  ! -iname "*bootstrap*" \
  ! -iname "*perm*" \
  ! -iname "*tmp*" \
  -print0 | while IFS= read -r -d '' file; do
    cp -f "$file" "$TABLE_DIR/"
done

#############################################
# 3. Collect Raw Statistical Artifacts
#############################################

if [ "$INCLUDE_ARTIFACTS" = true ]; then
    echo "Collecting statistical artifacts..."
    find "$RESULTS_DIR" \
      -type f \
      \( -iname "*bootstrap*.tsv" -o -iname "*perm*.tsv" -o -iname "*.json" \) \
      -print0 | while IFS= read -r -d '' file; do
        cp -f "$file" "$ARTIFACT_DIR/"
    done
fi

#############################################
# 4. Create Manifest
#############################################

echo "Creating manifest..."

MANIFEST="${PAPER_DIR}/MANIFEST_publication_assets.txt"

{
  echo "Publication Asset Manifest"
  echo "Generated: $(date)"
  echo ""
  echo "FIGURES:"
  ls -1 "$FIG_DIR" || true
  echo ""
  echo "TABLES:"
  ls -1 "$TABLE_DIR" || true
  echo ""
  if [ "$INCLUDE_ARTIFACTS" = true ]; then
      echo "ARTIFACTS:"
      ls -1 "$ARTIFACT_DIR" || true
  fi
} > "$MANIFEST"

echo "Done."
echo "Assets copied to:"
echo "  $FIG_DIR"
echo "  $TABLE_DIR"
echo "  $ARTIFACT_DIR"
echo "Manifest: $MANIFEST"

