SHELL := /bin/bash
.ONESHELL:

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
	$(PY) scripts/01_geo_download.py --config configs/run.yaml --outdir work/geo

qc:
	$(PY) scripts/02_load_and_qc.py --config configs/run.yaml --indir work/geo --outdir work/qc

annotate:
	$(PY) scripts/03_annotate_bt.py --config configs/run.yaml --indir work/qc --outdir work/annot

score:
	$(PY) scripts/04_score_programs.py --config configs/run.yaml --indir work/annot --outdir work/scored

liana:
	$(PY) scripts/05_run_liana.py --config configs/run.yaml --indir work/scored --outdir results/liana

report:
	$(PY) scripts/06_reports.py --indir results/liana --outdir results

lintfig:
	$(PY) scripts/99_check_no_png_only.py

# -----------------------------
# Compare LIANA datasets
# -----------------------------

COMPARE_OUTDIR = results/tables

compare: $(COMPARE_OUTDIR)/liana_overlap_summary.json

$(COMPARE_OUTDIR)/liana_overlap_summary.json: \
	results/liana/ssc_pbmc_gse195452.liana.tsv \
	results/liana/ssc_pbmc_gse210395.liana.tsv \
	scripts/07_compare_liana_datasets.py
	@echo "Running cross-dataset LIANA comparison..."
	@mkdir -p $(COMPARE_OUTDIR)
	python scripts/07_compare_liana_datasets.py \
		--a-name GSE195452 \
		--a results/liana/ssc_pbmc_gse195452.liana.tsv \
		--b-name GSE210395 \
		--b results/liana/ssc_pbmc_gse210395.liana.tsv \
		--sender B_cell \
		--receiver T_cell \
		--top-n 50 \
		--key lr \
		--outdir $(COMPARE_OUTDIR)
	@echo "Comparison complete."


all: dirs geo qc annotate score liana report

