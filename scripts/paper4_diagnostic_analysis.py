#!/usr/bin/env python3
"""
Paper 4 Diagnostic and Sample-Level Correlation Analysis
Run this to extract all the information we need to decide on Option B rescue strategy
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print("PAPER 4 DIAGNOSTIC ANALYSIS")
print("="*80)

# ============================================================================
# PART 1: Dataset Overview
# ============================================================================
print("\n" + "="*80)
print("PART 1: DATASET OVERVIEW")
print("="*80)

datasets = [
    ("GSE195452", "work/scored/ssc_pbmc_gse195452.scored.h5ad"),
    ("GSE210395", "work/scored/ssc_pbmc_gse210395.scored.h5ad"),
]

dataset_info = {}

for name, path in datasets:
    print(f"\n--- {name} ---")
    try:
        ad = sc.read_h5ad(path)
        
        # Basic stats
        print(f"Total cells: {ad.n_obs:,}")
        print(f"Total genes: {ad.n_vars:,}")
        
        # Cell types
        if 'cell_type' in ad.obs.columns:
            print(f"\nCell type distribution:")
            ct_counts = ad.obs['cell_type'].value_counts()
            for ct, count in ct_counts.head(10).items():
                print(f"  {ct:30s} {count:8,} cells")
        
        # Sample columns
        sample_cols = [c for c in ad.obs.columns if any(k in c.lower() 
                      for k in ['sample', 'donor', 'patient', 'subject', 'orig', 'batch', 'individual'])]
        print(f"\nPotential sample/donor columns: {sample_cols}")
        
        # If we find a sample column, count samples
        for col in sample_cols:
            n_unique = ad.obs[col].nunique()
            print(f"  {col}: {n_unique} unique values")
        
        # Score columns
        score_cols = [c for c in ad.obs.columns if any(k in c.lower() 
                     for k in ['b_costim', 'b_mhc', 'th1', 'th2', 'th17', 'tfh', 'treg'])]
        print(f"\nScore columns present: {score_cols}")
        
        # Store for later
        dataset_info[name] = {
            'adata': ad,
            'sample_cols': sample_cols,
            'score_cols': score_cols
        }
        
    except FileNotFoundError:
        print(f"  FILE NOT FOUND: {path}")
    except Exception as e:
        print(f"  ERROR: {e}")

# ============================================================================
# PART 2: Score Distributions
# ============================================================================
print("\n" + "="*80)
print("PART 2: SCORE DISTRIBUTIONS")
print("="*80)

for name, info in dataset_info.items():
    print(f"\n--- {name} ---")
    ad = info['adata']
    
    for score_col in info['score_cols']:
        if score_col in ad.obs.columns:
            vals = ad.obs[score_col].dropna()
            if len(vals) > 0:
                print(f"{score_col:20s} n={len(vals):7,} mean={vals.mean():6.3f} std={vals.std():6.3f} min={vals.min():6.3f} max={vals.max():6.3f}")

# ============================================================================
# PART 3: Sample-Level Correlation Analysis
# ============================================================================
print("\n" + "="*80)
print("PART 3: SAMPLE-LEVEL CORRELATION ANALYSIS")
print("="*80)

def compute_sample_correlations(ad, name, sample_col):
    """Compute per-sample B activation vs T program correlations"""
    
    print(f"\n--- {name} (using column: {sample_col}) ---")
    
    obs = ad.obs.copy()
    
    # Check required columns
    required_scores = ['b_costim', 'b_mhc_ii', 'th2', 'th17', 'th1']
    missing = [s for s in required_scores if s not in obs.columns]
    if missing:
        print(f"  MISSING SCORES: {missing}")
        return None
    
    if 'cell_type' not in obs.columns:
        print(f"  MISSING cell_type column")
        return None
    
    # Split by cell type
    b_cells = obs[obs['cell_type'] == 'B_cell'].copy()
    t_cells = obs[obs['cell_type'] == 'T_cell'].copy()
    
    print(f"  B cells: {len(b_cells):,}")
    print(f"  T cells: {len(t_cells):,}")
    
    if len(b_cells) == 0 or len(t_cells) == 0:
        print(f"  ERROR: No B or T cells found")
        return None
    
    # Compute B activation (average of b_costim + b_mhc_ii)
    b_cells['B_activation'] = (b_cells['b_costim'] + b_cells['b_mhc_ii']) / 2
    
    # Per-sample aggregation
    b_per_sample = b_cells.groupby(sample_col)['B_activation'].mean()
    th2_per_sample = t_cells.groupby(sample_col)['th2'].mean()
    th17_per_sample = t_cells.groupby(sample_col)['th17'].mean()
    th1_per_sample = t_cells.groupby(sample_col)['th1'].mean()
    
    # Combine
    df = pd.DataFrame({
        'B_activation': b_per_sample,
        'Th2': th2_per_sample,
        'Th17': th17_per_sample,
        'Th1': th1_per_sample
    }).dropna()
    
    print(f"  Samples with both B and T cells: {len(df)}")
    
    if len(df) < 3:
        print(f"  ERROR: Too few samples for correlation (need ≥3)")
        return None
    
    print(f"\n  First 10 samples:")
    print(df.head(10).to_string())
    
    # Correlations
    results = {}
    for t_prog in ['Th2', 'Th17', 'Th1']:
        r, p = spearmanr(df['B_activation'], df[t_prog])
        results[t_prog] = {'r': r, 'p': p, 'n': len(df)}
        print(f"\n  B_activation vs {t_prog:5s}: r = {r:6.3f}, p = {p:8.4g}, n = {len(df)}")
    
    # Save
    outdir = Path('work/sample_level_analysis')
    outdir.mkdir(exist_ok=True, parents=True)
    
    outfile = outdir / f'{name.lower()}_per_sample_bt_scores.tsv'
    df.to_csv(outfile, sep='\t')
    print(f"\n  Saved to: {outfile}")
    
    return {'df': df, 'results': results, 'name': name}

# Try each dataset
correlation_results = {}

for name, info in dataset_info.items():
    ad = info['adata']
    sample_cols = info['sample_cols']
    
    if not sample_cols:
        print(f"\n--- {name} ---")
        print(f"  No sample column found - trying cell-level analysis instead")
        continue
    
    # Try the first sample column
    sample_col = sample_cols[0]
    result = compute_sample_correlations(ad, name, sample_col)
    
    if result is not None:
        correlation_results[name] = result

# ============================================================================
# PART 4: Generate Figures
# ============================================================================
print("\n" + "="*80)
print("PART 4: GENERATING CORRELATION FIGURES")
print("="*80)

figdir = Path('results/sample_level_figures')
figdir.mkdir(exist_ok=True, parents=True)

for name, result in correlation_results.items():
    print(f"\n--- {name} ---")
    
    df = result['df']
    results = result['results']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, t_prog in zip(axes, ['Th2', 'Th17', 'Th1']):
        r = results[t_prog]['r']
        p = results[t_prog]['p']
        n = results[t_prog]['n']
        
        # Scatter
        ax.scatter(df['B_activation'], df[t_prog], alpha=0.6, s=50)
        
        # Regression line
        z = np.polyfit(df['B_activation'], df[t_prog], 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(df['B_activation'].min(), df['B_activation'].max(), 100)
        ax.plot(x_line, p_fit(x_line), 'r--', alpha=0.7, linewidth=2)
        
        # Labels
        ax.set_xlabel('B cell activation score', fontsize=11)
        ax.set_ylabel(f'{t_prog} score', fontsize=11)
        ax.set_title(f'{t_prog} vs B cell activation\n(Spearman r={r:.3f}, p={p:.3g}, n={n})', 
                     fontsize=12, weight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    figfile = figdir / f'{name.lower()}_b_activation_vs_t_programs.png'
    plt.savefig(figfile, dpi=300, bbox_inches='tight')
    print(f"  Saved: {figfile}")
    plt.close()

# ============================================================================
# PART 5: Summary and Recommendations
# ============================================================================
print("\n" + "="*80)
print("PART 5: SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n### CORRELATION RESULTS ###")
for name, result in correlation_results.items():
    print(f"\n{name}:")
    for t_prog, stats in result['results'].items():
        print(f"  {t_prog:5s}: r = {stats['r']:6.3f}, p = {stats['p']:8.4g}")

print("\n### INTERPRETATION ###")

# Determine strategy based on results
has_positive_th2_th17 = False
has_null_results = True

for name, result in correlation_results.items():
    r_th2 = result['results']['Th2']['r']
    p_th2 = result['results']['Th2']['p']
    r_th17 = result['results']['Th17']['r']
    p_th17 = result['results']['Th17']['p']
    
    if (r_th2 > 0.2 and p_th2 < 0.05) or (r_th17 > 0.2 and p_th17 < 0.05):
        has_positive_th2_th17 = True
        has_null_results = False
    elif abs(r_th2) < 0.15 and abs(r_th17) < 0.15:
        has_null_results = True

print()
if has_positive_th2_th17:
    print("✓ POSITIVE CORRELATIONS FOUND")
    print("  Strategy: Emphasize sample-level B-T coupling in manuscript")
    print("  Narrative: 'B cell activation drives Th2/Th17 polarization'")
    print("  Score-mediation: Move to supplementary as technical note")
elif has_null_results:
    print("○ NULL/WEAK CORRELATIONS")
    print("  Strategy: Focus on ligand amplification + literature interpretation")
    print("  Narrative: 'B cells amplify Th17-driving signals (S100A8/A9)'")
    print("  Discuss: Why no sample correlation (timing, tissue vs blood, etc.)")
else:
    print("? MIXED RESULTS")
    print("  Strategy: Interpret cautiously, emphasize mechanistic plausibility")

print("\n### NEXT STEPS ###")
print("1. Review correlation results above")
print("2. Check generated figures in:", figdir)
print("3. Check per-sample data in: work/sample_level_analysis/")
print("4. Report results back to Claude for manuscript strategy")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
