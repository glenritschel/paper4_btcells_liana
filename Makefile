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
	
	mkdir -p configs scripts notebooks data work \
		work/geo work/qc work/annot work/scored work/scored_q1q4 \
		results results/figures results/tables results/liana results/liana_q1q4 \
		results/compare results/compare_q1q4_195452 results/compare_q1q4_210395 \
		results/amplification_q1q4_195452 results/amplification_q1q4_210395
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

split_195452_q1q4:
	$(PY) scripts/08_make_quartile_bt_subsets.py \
		--indir work/scored \
		--outdir work/scored_q1q4 \
		--score-col b_costim \
		--min-b-cells 50 \
		--min-t-cells 50

split_210395_blow_bhigh:
	$(PY) scripts/08b_make_within_dataset_bt_subsets.py \
		--input work/scored/ssc_pbmc_gse210395.scored.h5ad \
		--outdir work/scored_q1q4 \
		--label ssc_pbmc_gse210395 \
		--groupby cell_type \
		--b-label B_cell \
		--t-label T_cell \
		--score-col b_costim \
		--low-quantile 0.25 \
		--high-quantile 0.75

liana_q1q4:
	$(PY) scripts/05_run_liana.py \
		--config configs/run.yaml \
		--indir work/scored_q1q4 \
		--outdir results/liana_q1q4

compare_195452_q1q4:
	$(PY) scripts/07_compare_liana_datasets.py \
		--a results/liana_q1q4/ssc_pbmc_gse195452.BT.q1.liana_top_bt.tsv \
		--b results/liana_q1q4/ssc_pbmc_gse195452.BT.q4.liana_top_bt.tsv \
		--outdir results/compare_q1q4_195452 \
		--topk 0 \
		--label-a gse195452_Q1 \
		--label-b gse195452_Q4 \
		--universe full_union \
		--perm-reps 20000

compare_210395_blow_bhigh:
	$(PY) scripts/07_compare_liana_datasets.py \
		--a results/liana_q1q4/ssc_pbmc_gse210395.BT.Blow.liana_top_bt.tsv \
		--b results/liana_q1q4/ssc_pbmc_gse210395.BT.Bhigh.liana_top_bt.tsv \
		--outdir results/compare_q1q4_210395 \
		--topk 0 \
		--label-a gse210395_Blow \
		--label-b gse210395_Bhigh \
		--universe full_union \
		--perm-reps 20000

amp_195452_q1q4:
	$(PY) scripts/09_lr_amplification.py \
		--adata-a work/scored_q1q4/ssc_pbmc_gse195452.BT.q1.scored.h5ad \
		--adata-b work/scored_q1q4/ssc_pbmc_gse195452.BT.q4.scored.h5ad \
		--overlap results/compare_q1q4_195452/compare_gse195452_Q1_vs_gse195452_Q4_topALL.overlap_by_delta.tsv \
		--outdir results/amplification_q1q4_195452 \
		--label-a gse195452_Q1 \
		--label-b gse195452_Q4 \
		--focus BtoT

amp_210395_blow_bhigh:
	$(PY) scripts/09_lr_amplification.py \
		--adata-a work/scored_q1q4/ssc_pbmc_gse210395.BT.Blow.scored.h5ad \
		--adata-b work/scored_q1q4/ssc_pbmc_gse210395.BT.Bhigh.scored.h5ad \
		--overlap results/compare_q1q4_210395/compare_gse210395_Blow_vs_gse210395_Bhigh_topALL.overlap_by_delta.tsv \
		--outdir results/amplification_q1q4_210395 \
		--label-a gse210395_Blow \
		--label-b gse210395_Bhigh \
		--focus BtoT

corr_195452_q1q4:
	$(PY) scripts/spearman_correlation.py \
		--input results/amplification_q1q4_195452/lr_amplification_gse195452_Q1_vs_gse195452_Q4.tsv

corr_210395_blow_bhigh:
	$(PY) scripts/spearman_correlation.py \
		--input results/amplification_q1q4_210395/lr_amplification_gse210395_Blow_vs_gse210395_Bhigh.tsv

pcorr_195452_q1q4:
	$(PY) scripts/partial_correlation.py \
		--input results/amplification_q1q4_195452/lr_amplification_gse195452_Q1_vs_gse195452_Q4.tsv \
		--controls lig_log2fc
	$(PY) scripts/partial_correlation.py \
		--input results/amplification_q1q4_195452/lr_amplification_gse195452_Q1_vs_gse195452_Q4.tsv \
		--controls rec_log2fc
	$(PY) scripts/partial_correlation.py \
		--input results/amplification_q1q4_195452/lr_amplification_gse195452_Q1_vs_gse195452_Q4.tsv \
		--controls lig_log2fc rec_log2fc

pcorr_210395_blow_bhigh:
	$(PY) scripts/partial_correlation.py \
		--input results/amplification_q1q4_210395/lr_amplification_gse210395_Blow_vs_gse210395_Bhigh.tsv \
		--controls lig_log2fc
	$(PY) scripts/partial_correlation.py \
		--input results/amplification_q1q4_210395/lr_amplification_gse210395_Blow_vs_gse210395_Bhigh.tsv \
		--controls rec_log2fc
	$(PY) scripts/partial_correlation.py \
		--input results/amplification_q1q4_210395/lr_amplification_gse210395_Blow_vs_gse210395_Bhigh.tsv \
		--controls lig_log2fc rec_log2fc

q1q4: split_195452_q1q4 split_210395_blow_bhigh liana_q1q4 \
      compare_195452_q1q4 compare_210395_blow_bhigh \
      amp_195452_q1q4 amp_210395_blow_bhigh \
      corr_195452_q1q4 corr_210395_blow_bhigh \
      pcorr_195452_q1q4 pcorr_210395_blow_bhigh


# -----------------------------
# Compare LIANA datasets
# -----------------------------
COMPARE_OUTDIR = results/compare

compare: $(COMPARE_OUTDIR)/compare_gse195452_vs_gse210395_topALL.summary.json

$(COMPARE_OUTDIR)/compare_gse195452_vs_gse210395_topALL.summary.json: \
	results/liana/ssc_pbmc_gse195452.BT.liana_top_bt.tsv \
	results/liana/ssc_pbmc_gse210395.BT.liana_top_bt.tsv \
	results/liana/ssc_pbmc_gse195452.BT.liana.tsv \
	results/liana/ssc_pbmc_gse210395.BT.liana.tsv \
	scripts/07_compare_liana_datasets.py
	@echo "Running cross-dataset LIANA comparison..."
	@mkdir -p $(COMPARE_OUTDIR)
	$(PY) scripts/07_compare_liana_datasets.py \
		--a results/liana/ssc_pbmc_gse195452.BT.liana_top_bt.tsv \
		--b results/liana/ssc_pbmc_gse210395.BT.liana_top_bt.tsv \
		--a-full results/liana/ssc_pbmc_gse195452.BT.liana.tsv \
		--b-full results/liana/ssc_pbmc_gse210395.BT.liana.tsv \
		--outdir $(COMPARE_OUTDIR) \
		--topk 0 \
		--label-a gse195452 \
		--label-b gse210395 \
		--universe full_union \
		--perm-reps 20000 \
		--seed 0
	@echo "Comparison complete."


all: dirs geo qc annotate score liana report

