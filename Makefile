SHELL := /bin/bash

ENV ?= btcells-scvi
PY  ?= python

.PHONY: help env check dirs geo qc annotate score liana report all

help:
	@echo "Targets:"
	@echo "  env       - print env info"
	@echo "  check     - import checks"
	@echo "  dirs      - create standard dirs"
	@echo "  geo       - download GEO datasets listed in configs/datasets.yaml"
	@echo "  qc        - load + QC + save work/*.h5ad"
	@echo "  annotate  - annotate B/T + save annotated h5ad"
	@echo "  score     - score signatures/programs"
	@echo "  liana     - run LIANA B<->T interactions"
	@echo "  report    - generate summary tables/figures"
	@echo "  all       - run full pipeline"

env:
	$(PY) scripts/00_print_env.py

check:
	$(PY) -c "import scanpy, anndata, pandas, numpy; import liana; print('OK')"

dirs:
	mkdir -p configs scripts notebooks data work results results/figures results/tables results/liana
	touch data/README.md work/README.md results/README.md notebooks/README.md

geo:
	$(PY) scripts/01_geo_download.py --config configs/datasets.yaml --outdir work/geo

qc:
	$(PY) scripts/02_load_and_qc.py --config configs/qc.yaml --indir work/geo --outdir work/qc

annotate:
	$(PY) scripts/03_annotate_bt.py --config configs/qc.yaml --indir work/qc --outdir work/annot

score:
	$(PY) scripts/04_score_programs.py --config configs/scoring.yaml --indir work/annot --outdir work/scored

liana:
	$(PY) scripts/05_run_liana.py --config configs/liana.yaml --indir work/scored --outdir results/liana

report:
	$(PY) scripts/06_reports.py --indir results/liana --outdir results

all: dirs geo qc annotate score liana report

