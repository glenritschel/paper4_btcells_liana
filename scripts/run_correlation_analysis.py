#!/usr/bin/env python3
"""
Paper 4 Sample-Level Correlation Analysis (without th1)
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("SAMPLE-LEVEL CORRELATION ANALYSIS - GSE195452")
print("="*80)

# Load data
ad = sc.read_h5ad("work/scored/ssc_pbmc_gse195452.scored.h5ad")
print(f"\nTotal cells: {ad.n_obs:,}")

obs = ad.obs.copy()

# Split by cell type
b_cells = obs[obs['cell_type'] == 'B_cell'].copy()
t_cells = obs[obs['cell_type'] == 'T_cell'].copy()

print(f"B cells: {len(b_cells):,}")
print(f"T cells: {len(t_cells):,}")

# Compute B activation (average of b_costim + b_mhc_ii)
b_cells['B_activation'] = (b_cells['b_costim'] + b_cells['b_mhc_ii']) / 2

# Per-sample aggregation
print("\nComputing per-sample means...")
b_per_sample = b_cells.groupby('sample_id')['B_activation'].mean()
th2_per_sample = t_cells.groupby('sample_id')['th2'].mean()
th17_per_sample = t_cells.groupby('sample_id')['th17'].mean()

# Count cells per sample
b_counts = b_cells.groupby('sample_id').size()
t_counts = t_cells.groupby('sample_id').size()

# Combine
df = pd.DataFrame({
    'B_activation': b_per_sample,
    'Th2': th2_per_sample,
    'Th17': th17_per_sample,
    'n_B_cells': b_counts,
    'n_T_cells': t_counts
}).dropna()

print(f"Samples with both B and T cells: {len(df)}")
print(f"\nSample statistics:")
print(f"  B cells per sample: median={df['n_B_cells'].median():.0f}, range={df['n_B_cells'].min():.0f}-{df['n_B_cells'].max():.0f}")
print(f"  T cells per sample: median={df['n_T_cells'].median():.0f}, range={df['n_T_cells'].min():.0f}-{df['n_T_cells'].max():.0f}")

print(f"\nFirst 10 samples:")
print(df.head(10).to_string())

# Correlations
print("\n" + "="*80)
print("CORRELATION RESULTS")
print("="*80)

r_th2, p_th2 = spearmanr(df['B_activation'], df['Th2'])
r_th17, p_th17 = spearmanr(df['B_activation'], df['Th17'])

print(f"\nB_activation vs Th2:  r = {r_th2:7.4f}, p = {p_th2:10.6g}, n = {len(df)}")
print(f"B_activation vs Th17: r = {r_th17:7.4f}, p = {p_th17:10.6g}, n = {len(df)}")

# Interpretation
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

def interpret_correlation(r, p, threshold_r=0.15, threshold_p=0.05):
    if p < threshold_p:
        if r > threshold_r:
            return "POSITIVE - statistically significant"
        elif r < -threshold_r:
            return "NEGATIVE - statistically significant"
        else:
            return "WEAK but significant"
    else:
        return "NULL - not significant"

print(f"\nTh2:  {interpret_correlation(r_th2, p_th2)}")
print(f"Th17: {interpret_correlation(r_th17, p_th17)}")

# Save results
outdir = Path('work/sample_level_analysis')
outdir.mkdir(exist_ok=True, parents=True)

outfile = outdir / 'gse195452_per_sample_bt_scores.tsv'
df.to_csv(outfile, sep='\t')
print(f"\nSaved per-sample data to: {outfile}")

# Generate figure
print("\n" + "="*80)
print("GENERATING FIGURE")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, t_prog, r, p in zip(axes, ['Th2', 'Th17'], [r_th2, r_th17], [p_th2, p_th17]):
    # Scatter
    ax.scatter(df['B_activation'], df[t_prog], alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    # Regression line
    z = np.polyfit(df['B_activation'], df[t_prog], 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(df['B_activation'].min(), df['B_activation'].max(), 100)
    ax.plot(x_line, p_fit(x_line), 'r--', alpha=0.7, linewidth=2)
    
    # Labels
    ax.set_xlabel('B cell activation score', fontsize=12)
    ax.set_ylabel(f'{t_prog} score', fontsize=12)
    ax.set_title(f'{t_prog} vs B cell activation\nSpearman r={r:.3f}, p={p:.4g}, n={len(df)}', 
                 fontsize=13, weight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add significance marker
    if p < 0.05:
        ax.text(0.05, 0.95, '**' if p < 0.01 else '*', 
                transform=ax.transAxes, fontsize=20, 
                verticalalignment='top', color='red', weight='bold')

plt.tight_layout()

figdir = Path('results/sample_level_figures')
figdir.mkdir(exist_ok=True, parents=True)

figfile = figdir / 'gse195452_b_activation_vs_t_programs.png'
plt.savefig(figfile, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {figfile}")

# Also save as PDF for manuscript
figfile_pdf = figdir / 'gse195452_b_activation_vs_t_programs.pdf'
plt.savefig(figfile_pdf, bbox_inches='tight')
print(f"Saved PDF to: {figfile_pdf}")

plt.close()

# Summary statistics for manuscript
print("\n" + "="*80)
print("SUMMARY STATISTICS FOR MANUSCRIPT")
print("="*80)

print(f"""
Dataset: GSE195452 (SSc PBMCs)
Samples analyzed: {len(df)}
B cells: {len(b_cells):,} total across samples
T cells: {len(t_cells):,} total across samples

Correlation Results:
- B cell activation vs Th2:  Spearman r = {r_th2:.3f}, p = {p_th2:.4g}
- B cell activation vs Th17: Spearman r = {r_th17:.3f}, p = {p_th17:.4g}
""")

print("\n" + "="*80)
print("RECOMMENDATION FOR MANUSCRIPT")
print("="*80)

if (abs(r_th2) > 0.15 and p_th2 < 0.05) or (abs(r_th17) > 0.15 and p_th17 < 0.05):
    print("""
✓ SIGNIFICANT CORRELATIONS FOUND

Manuscript Strategy:
1. Lead with sample-level correlation finding
2. Emphasize biological coupling between B activation and T polarization
3. Use S100A8/A9 as mechanistic explanation
4. Move score-mediation to supplementary material

Key Message: "B cell activation correlates with Th2/Th17 polarization at the 
sample level, mediated by amplification of inflammatory ligands"
""")
else:
    print("""
○ NO SIGNIFICANT SAMPLE-LEVEL CORRELATIONS

Manuscript Strategy:
1. Lead with ligand amplification finding (S100A8/A9)
2. Connect to Th17 via literature (TLR4 mechanism)
3. Discuss why sample correlation may be absent:
   - Temporal dynamics (B activation precedes T polarization)
   - Spatial compartmentalization (blood vs tissue)
   - Threshold effects (need sustained activation)
4. Frame as hypothesis-generating for spatial/functional validation

Key Message: "B cell activation amplifies expression of Th17-polarizing ligands 
(S100A8/A9), providing mechanistic basis for B-T-fibroblast pathway"
""")

print("\nDONE - Report results to Claude")
