#!/usr/bin/env python3
"""
Generate All Publication Figures from Pipeline Results
Paper 4: B-T Cell Interactions in Systemic Sclerosis

This script generates all figures from pipeline results in their final publication format.
Requires completed pipeline run with all results/ and work/ files present.

Author: Glen Ritschel
Date: February 13, 2026
Updated: February 14, 2026 - Fixed to use actual data column names
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import scanpy as sc
import shutil

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
sns.set_style("ticks")

# Paths
REPO_ROOT = Path.cwd()
WORK_DIR = REPO_ROOT / "work"
RESULTS_DIR = REPO_ROOT / "results"
PAPER_FIGURES = REPO_ROOT / "paper" / "figures"
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)


def generate_umaps():
    """
    Generate UMAP plots for both datasets
    Uses the same approach as ChatGPT: compute UMAP if needed, plot B cells only
    """
    print("Generating UMAPs for B cell activation...")
    
    datasets = [
        {
            'name': 'GSE195452',
            'h5ad': WORK_DIR / "scored_bt" / "ssc_pbmc_gse195452.BT.scored.h5ad",
            'output_base': 'Fig1A1_B_activation_umap_gse195452'
        },
        {
            'name': 'GSE210395', 
            'h5ad': WORK_DIR / "scored_bt" / "ssc_pbmc_gse210395.BT.scored.h5ad",
            'output_base': 'Fig1A2_B_activation_umap_gse210395'
        }
    ]
    
    for dataset in datasets:
        try:
            print(f"  Processing {dataset['name']}...")
            
            # Load data
            adata = sc.read_h5ad(dataset['h5ad'])
            
            # Filter to B cells only
            b_cells = adata[adata.obs['cell_type'] == 'B_cell'].copy()
            
            print(f"    Found {len(b_cells)} B cells")
            
            # Compute UMAP if not present
            if 'X_umap' not in b_cells.obsm:
                print(f"    Computing UMAP...")
                sc.pp.neighbors(b_cells, n_neighbors=15, n_pcs=30)
                sc.tl.umap(b_cells)
            
            # Check for score column (try different possible names)
            score_col = None
            for col in ['b_activation', 'B_activation_combined', 'b_activation_combined']:
                if col in b_cells.obs.columns:
                    score_col = col
                    break
            
            if score_col is None:
                print(f"    Warning: No B activation score column found")
                print(f"    Available columns: {list(b_cells.obs.columns)}")
                continue
            
            print(f"    Using score column: {score_col}")
            
            # Plot UMAP
            fig, ax = plt.subplots(figsize=(8, 6))
            sc.pl.umap(b_cells, color=score_col, ax=ax, show=False, 
                      title=f'{dataset["name"]} B Cell Activation',
                      cmap='viridis', frameon=False)
            
            # Save
            output_png = PAPER_FIGURES / f"{dataset['output_base']}.png"
            output_pdf = PAPER_FIGURES / f"{dataset['output_base']}.pdf"
            
            plt.tight_layout()
            plt.savefig(output_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_pdf, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ Saved: {output_png}")
            print(f"    ✓ Saved: {output_pdf}")
            
        except Exception as e:
            print(f"    Error processing {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()


def generate_figure1():
    """
    Figure 1: B Cell Activation Stratification
    Panels: A1 (GSE195452 UMAP), A2 (GSE210395 UMAP), B (Violin plots), C (Distribution)
    """
    print("Generating Figure 1: B Cell Activation Stratification...")
    
    # First generate the individual UMAPs
    generate_umaps()
    
    # Now create composite figure
    fig = plt.figure(figsize=(16, 4.5))
    
    # Panel A1: GSE195452 UMAP
    ax1 = plt.subplot(141)
    umap1 = PAPER_FIGURES / "Fig1A1_B_activation_umap_gse195452.png"
    if umap1.exists():
        from PIL import Image
        img = Image.open(umap1)
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('A1. B Cell Activation (GSE195452)', fontweight='bold', loc='left')
    else:
        ax1.text(0.5, 0.5, 'GSE195452 UMAP\n(Run failed)', ha='center', va='center')
        ax1.set_title('A1. B Cell Activation (GSE195452)', fontweight='bold', loc='left')
    
    # Panel A2: GSE210395 UMAP
    ax2 = plt.subplot(142)
    umap2 = PAPER_FIGURES / "Fig1A2_B_activation_umap_gse210395.png"
    if umap2.exists():
        from PIL import Image
        img = Image.open(umap2)
        ax2.imshow(img)
        ax2.axis('off')
        ax2.set_title('A2. B Cell Activation (GSE210395)', fontweight='bold', loc='left')
    else:
        ax2.text(0.5, 0.5, 'GSE210395 UMAP\n(Run failed)', ha='center', va='center')
        ax2.set_title('A2. B Cell Activation (GSE210395)', fontweight='bold', loc='left')
    
    # Panel B: Violin plots of key markers (Q1 vs Q4)
    ax3 = plt.subplot(143)
    
    try:
        q1 = sc.read_h5ad(WORK_DIR / "scored_q1q4" / "ssc_pbmc_gse195452.BT.q1.scored.h5ad")
        q4 = sc.read_h5ad(WORK_DIR / "scored_q1q4" / "ssc_pbmc_gse195452.BT.q4.scored.h5ad")
        
        # Filter to B cells
        q1_b = q1[q1.obs['cell_type'] == 'B_cell']
        q4_b = q4[q4.obs['cell_type'] == 'B_cell']
        
        # Extract expression for key genes
        genes = ['HLA-DRA', 'CD80', 'CD86']
        q1_data = [q1_b[:, gene].X.toarray().flatten() for gene in genes]
        q4_data = [q4_b[:, gene].X.toarray().flatten() for gene in genes]
        
        # Violin plots
        positions_q1 = [0.8, 2.8, 4.8]
        positions_q4 = [1.2, 3.2, 5.2]
        
        parts1 = ax3.violinplot(q1_data, positions=positions_q1, widths=0.35,
                                showmeans=True, showextrema=True)
        for pc in parts1['bodies']:
            pc.set_facecolor('#6baed6')
            pc.set_alpha(0.7)
        
        parts2 = ax3.violinplot(q4_data, positions=positions_q4, widths=0.35,
                                showmeans=True, showextrema=True)
        for pc in parts2['bodies']:
            pc.set_facecolor('#fd8d3c')
            pc.set_alpha(0.7)
        
        ax3.set_xticks([1, 3, 5])
        ax3.set_xticklabels(genes)
        ax3.set_ylabel('Normalized Expression')
        ax3.set_title('B. Key Activation Markers', fontweight='bold', loc='left')
        ax3.legend([mpatches.Patch(color='#6baed6'), mpatches.Patch(color='#fd8d3c')],
                   ['Q1 (Resting)', 'Q4 (Activated)'], loc='upper left', frameon=False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
    except Exception as e:
        print(f"  Warning: Could not create violin plots: {e}")
        ax3.text(0.5, 0.5, 'Violin plots\n(Load actual data)', ha='center', va='center')
        ax3.set_title('B. Key Activation Markers', fontweight='bold', loc='left')
    
    # Panel C: Per-sample distribution
    ax4 = plt.subplot(144)
    
    try:
        df = pd.read_csv(WORK_DIR / "sample_level_analysis" / "gse195452_per_sample_bt_scores.tsv", sep='\t')
        
        # Try different possible column names
        col_name = None
        for name in ['b_activation_mean', 'B_activation_mean', 'b_activation', 'B_activation_combined']:
            if name in df.columns:
                col_name = name
                break
        
        if col_name is None:
            print(f"  Warning: No B activation column found. Available: {list(df.columns)}")
            raise ValueError("No B activation column")
        
        ax4.hist(df[col_name], bins=25, color='#3182bd', alpha=0.7, 
                edgecolor='black', linewidth=0.5)
        mean_val = df[col_name].mean()
        ax4.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean = {mean_val:.2f}', alpha=0.8)
        
        ax4.text(mean_val, ax4.get_ylim()[1] * 0.9, f'n = {len(df)} samples',
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))
        
        ax4.set_xlabel('B Cell Activation Score')
        ax4.set_ylabel('Number of Samples')
        ax4.set_title('C. Per-Sample Distribution', fontweight='bold', loc='left')
        ax4.legend(frameon=False)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
    except Exception as e:
        print(f"  Warning: Could not load per-sample data: {e}")
        ax4.text(0.5, 0.5, 'Distribution\n(Load actual data)', ha='center', va='center')
        ax4.set_title('C. Per-Sample Distribution', fontweight='bold', loc='left')
    
    plt.tight_layout()
    
    # Save composite
    output_png = PAPER_FIGURES / "Figure1_B_Cell_Activation.png"
    output_pdf = PAPER_FIGURES / "Figure1_B_Cell_Activation.pdf"
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved composite: {output_png}")
    print(f"  ✓ Saved composite: {output_pdf}")


def generate_figure2():
    """
    Figure 2: Ligand Amplification
    Quick solution: Copy from existing ChatGPT-generated figure in results/
    """
    print("Generating Figure 2: Ligand Amplification...")
    
    # Try to find the existing figure from ChatGPT analysis
    possible_sources = [
        RESULTS_DIR / "figures" / "gse210395_blow_vs_bhigh_rank_vs_amp.png",
        RESULTS_DIR / "figures" / "gse210395_blow_vs_bhigh_rank_vs_amp.pdf",
        REPO_ROOT / "paper" / "figures" / "Fig1_delta_rank_vs_amp_scatter_gse210395.png",
    ]
    
    copied = False
    for source in possible_sources:
        if source.exists():
            # Copy to Figure2 location
            if source.suffix == '.png':
                dest_png = PAPER_FIGURES / "Figure2_Ligand_Amplification.png"
                shutil.copy(source, dest_png)
                print(f"  ✓ Copied from: {source}")
                print(f"  ✓ To: {dest_png}")
                
                # Also create PDF if possible
                try:
                    from PIL import Image
                    img = Image.open(source)
                    dest_pdf = PAPER_FIGURES / "Figure2_Ligand_Amplification.pdf"
                    img.save(dest_pdf, 'PDF', resolution=300.0)
                    print(f"  ✓ Created: {dest_pdf}")
                except:
                    pass
                
                copied = True
                break
            
            elif source.suffix == '.pdf':
                dest_pdf = PAPER_FIGURES / "Figure2_Ligand_Amplification.pdf"
                shutil.copy(source, dest_pdf)
                print(f"  ✓ Copied from: {source}")
                print(f"  ✓ To: {dest_pdf}")
                copied = True
                break
    
    if not copied:
        print("  ! Could not find existing Figure 2 from ChatGPT analysis")
        print("    Searched in:")
        for source in possible_sources:
            print(f"      - {source}")
        print()
        print("  Manual solution:")
        print("    Look for rank_vs_amp figures in results/figures/")
        print("    Copy the best one to paper/figures/Figure2_Ligand_Amplification.png")


def generate_figure3():
    """
    Figure 3: Negative Correlations (KEY FIGURE - ACTUAL DATA)
    This should already exist from scripts/run_correlation_analysis.py
    Just verify and copy to paper/figures/
    """
    print("Generating Figure 3: Negative Correlations (ACTUAL DATA)...")
    
    source_png = RESULTS_DIR / "sample_level_figures" / "gse195452_b_activation_vs_t_programs.png"
    source_pdf = RESULTS_DIR / "sample_level_figures" / "gse195452_b_activation_vs_t_programs.pdf"
    
    if source_pdf.exists():
        dest_png = PAPER_FIGURES / "Figure3_Negative_Correlations.png"
        dest_pdf = PAPER_FIGURES / "Figure3_Negative_Correlations.pdf"
        
        if source_png.exists():
            shutil.copy(source_png, dest_png)
            print(f"  ✓ Copied: {dest_png}")
        
        shutil.copy(source_pdf, dest_pdf)
        print(f"  ✓ Copied: {dest_pdf}")
        print("  ℹ This figure contains Glen's actual discovery data!")
    else:
        print("  ! Figure 3 not found - run scripts/run_correlation_analysis.py first")
        print("    This is THE KEY FIGURE with actual statistics:")
        print("    - B vs Th2: r=-0.368, p=2.089e-10")
        print("    - B vs Th17: r=-0.297, p=3.995e-07")


def generate_figure4():
    """
    Figure 4: Mechanistic Model (Galectin-9-TIM-3)
    Conceptual diagram - no data required
    """
    print("Generating Figure 4: Mechanistic Model...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Activated B cell
    b_cell = mpatches.FancyBboxPatch((0.5, 3), 2, 1.5,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightblue',
                                     edgecolor='blue', linewidth=2)
    ax.add_patch(b_cell)
    ax.text(1.5, 3.75, 'Activated B Cell', ha='center', va='center',
           fontweight='bold', fontsize=11)
    ax.text(1.5, 3.3, 'MHC-II ↑\nCD80/CD86 ↑\nLGALS9 ↑', ha='center', va='center',
           fontsize=9, style='italic')
    
    # Galectin-9
    gal9 = mpatches.Circle((3.5, 3.75), 0.3, facecolor='yellow',
                          edgecolor='orange', linewidth=2)
    ax.add_patch(gal9)
    ax.text(3.5, 3.75, 'Gal-9', ha='center', va='center',
           fontsize=9, fontweight='bold')
    
    # Arrow B → Gal9
    ax.annotate('', xy=(3.2, 3.75), xytext=(2.5, 3.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    ax.text(2.85, 4.0, 'secrete/\nexpress', ha='center', fontsize=8, style='italic')
    
    # T cell
    t_cell = mpatches.FancyBboxPatch((5, 3), 2, 1.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor='lightcoral',
                                    edgecolor='red', linewidth=2)
    ax.add_patch(t_cell)
    ax.text(6, 3.75, 'T Cell', ha='center', va='center',
           fontweight='bold', fontsize=11)
    ax.text(6, 3.3, 'TIM-3 +', ha='center', va='center',
           fontsize=9, style='italic')
    
    # Arrow Gal9 → T
    ax.annotate('', xy=(5, 3.75), xytext=(3.8, 3.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(4.4, 4.0, 'binds\nTIM-3', ha='center', fontsize=8, style='italic')
    
    # Downstream effects
    ax.text(6, 2.3, '↓', ha='center', fontsize=20, color='red')
    ax.text(6, 1.9, 'Ca²⁺ influx', ha='center', fontsize=9)
    ax.text(6, 1.6, 'Caspase activation', ha='center', fontsize=9)
    ax.text(6, 1.3, 'T cell apoptosis', ha='center', fontsize=9, fontweight='bold')
    ax.text(6, 1.0, 'Functional exhaustion', ha='center', fontsize=9, fontweight='bold')
    
    # Result box
    result_box = mpatches.FancyBboxPatch((7.5, 0.3), 2, 1.2,
                                        boxstyle="round,pad=0.1",
                                        facecolor='lightyellow',
                                        edgecolor='black', linewidth=2)
    ax.add_patch(result_box)
    ax.text(8.5, 1.2, 'Result:', ha='center', fontweight='bold', fontsize=10)
    ax.text(8.5, 0.9, 'High B activation', ha='center', fontsize=9)
    ax.text(8.5, 0.7, '↓', ha='center', fontsize=12)
    ax.text(8.5, 0.5, 'Low Th2/Th17', ha='center', fontsize=9)
    
    # Title
    ax.text(5, 5.5, 'Galectin-9-TIM-3 Regulatory Axis', ha='center',
           fontsize=14, fontweight='bold')
    
    output_png = PAPER_FIGURES / "Figure4_Mechanistic_Model.png"
    output_pdf = PAPER_FIGURES / "Figure4_Mechanistic_Model.pdf"
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_png}")
    print(f"  ✓ Saved: {output_pdf}")


def generate_figureS1():
    """
    Figure S1: Rank vs Amplification (Partial Correlation)
    """
    print("Generating Figure S1: Rank vs Amplification...")
    
    # This should exist from ChatGPT analysis
    source = RESULTS_DIR / "robustness_gse210395" / "gse210395_delta_rank_vs_amp.scatter.png"
    
    if source.exists():
        dest_png = PAPER_FIGURES / "FigureS1_Rank_Amplification.png"
        shutil.copy(source, dest_png)
        print(f"  ✓ Copied: {dest_png}")
        
        # Also create PDF version
        try:
            from PIL import Image
            img = Image.open(source)
            dest_pdf = PAPER_FIGURES / "FigureS1_Rank_Amplification.pdf"
            img.save(dest_pdf, 'PDF', resolution=300.0)
            print(f"  ✓ Created: {dest_pdf}")
        except Exception as e:
            print(f"  Note: Could not create PDF: {e}")
    else:
        print("  ! Figure S1 source not found")
        print(f"    Expected: {source}")


def main():
    """Generate all publication figures"""
    print("="*70)
    print("GENERATING ALL PUBLICATION FIGURES")
    print("Paper 4: B-T Cell Interactions in Systemic Sclerosis")
    print("="*70)
    print()
    
    # Check required directories
    if not WORK_DIR.exists():
        print(f"Error: Work directory not found: {WORK_DIR}")
        print("Run the full pipeline first to generate all required data.")
        sys.exit(1)
    
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        print("Run the full pipeline first to generate all required data.")
        sys.exit(1)
    
    # Generate each figure
    try:
        generate_figure1()
        print()
    except Exception as e:
        print(f"Error generating Figure 1: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    try:
        generate_figure2()
        print()
    except Exception as e:
        print(f"Error generating Figure 2: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    try:
        generate_figure3()
        print()
    except Exception as e:
        print(f"Error generating Figure 3: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    try:
        generate_figure4()
        print()
    except Exception as e:
        print(f"Error generating Figure 4: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    try:
        generate_figureS1()
        print()
    except Exception as e:
        print(f"Error generating Figure S1: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    print("="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)
    print()
    print("Output directory:", PAPER_FIGURES)
    print()
    print("Generated files:")
    for f in sorted(PAPER_FIGURES.glob("Figure*.png")):
        print(f"  ✓ {f.name}")
    print()
    print("Next steps:")
    print("1. Review all figures in paper/figures/")
    print("2. Convert to JPG if needed for journal submission:")
    print("   cd paper/figures && for f in Figure*.png; do")
    print('     convert "$f" -density 300 -quality 95 "${f%.png}.jpg"')
    print("   done")
    print("3. Run scripts/publication/create_supplementary_tables.py")
    print("4. Proceed with GitHub Release and Zenodo archiving")


if __name__ == '__main__':
    main()

