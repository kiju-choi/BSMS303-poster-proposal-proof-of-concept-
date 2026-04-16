"""
01_preprocessing.py — Kidney organoid scRNA-seq preprocessing

Ungricht et al. (2022) kidney organoid 데이터 전처리.
목적: downstream OT trajectory inference를 위한 깨끗한 AnnData 객체 생성.
moscot이 의미 있는 transport map을 계산하려면 cell type annotation과
time label이 정확해야 하므로, QC → normalization → clustering → annotation
순서를 엄격히 따름.

Input:  data/raw/*.h5 (7 samples, 10X Cell Ranger filtered matrices)
Output: data/processed/adata_preprocessed.h5ad

전처리 파라미터는 원논문 Methods 기준,
Louvain → Leiden, regress_out 제거 등 합리적 범위에서 조정.
"""

import scanpy as sc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import doubletdetection
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
FIGURE_DIR = os.path.join(PROJECT_DIR, 'figures')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Sample metadata: filename -> (day, replicate)
# 3 timepoints (Day 21/28/35), 7 samples total — moscot의 time_key로 사용됨
SAMPLES = {
    'GSM5507784_3D_day21_1_filtered_gene_bc_matrices_h5.h5': (21, '1'),
    'GSM5507785_3D_day21_2_filtered_gene_bc_matrices_h5.h5': (21, '2'),
    'GSM5507786_3D_day28_1_filtered_gene_bc_matrices_h5.h5': (28, '1'),
    'GSM5507787_3D_day28_2_filtered_gene_bc_matrices_h5.h5': (28, '2'),
    'GSM5507788_3D_day28_3_filtered_gene_bc_matrices_h5.h5': (28, '3'),
    'GSM5507789_3D_day35_1_filtered_gene_bc_matrices_h5.h5': (35, '1'),
    'GSM5507790_3D_day35_2_filtered_gene_bc_matrices_h5.h5': (35, '2'),
}

# Marker genes for cell type annotation (Ungricht et al. Table S1A, Figure S1)
# 원논문에서 정의한 11 cell type 중 Muscle(off-target)은 소수라 별도 마커 불필요
MARKER_GENES = {
    'eTEC':      ['EPCAM', 'PAX8'],
    'PTEC':      ['CUBN', 'LRP2', 'HNF4A'],
    'LH':        ['SLC12A1'],
    'DTEC':      ['GATA3', 'POU3F3'],
    'Podocyte':  ['PODXL', 'NPHS1', 'NPHS2'],
    'eP':        ['PODXL', 'MAFB'],
    'Stroma':    ['COL1A1', 'PDGFRB', 'DCN'],
    'aStroma':   ['ACTA2', 'TAGLN'],
    'Cycling':   ['MKI67', 'TOP2A'],
    'Muscle':    ['MYOD1', 'MYLPF', 'TNNC2'],
}


# ---------------------------------------------------------------------------
# Step 1: Load and merge all samples
# ---------------------------------------------------------------------------
def load_and_merge_samples():
    """Load 7 H5 files, annotate with day/replicate, and concatenate."""
    print("=" * 60)
    print("Step 1: Loading and merging samples")
    print("=" * 60)

    adatas = []
    for filename, (day, rep) in SAMPLES.items():
        path = os.path.join(RAW_DIR, filename)
        adata = sc.read_10x_h5(path)
        adata.var_names_make_unique()

        # Annotate metadata
        sample_id = f"day{day}_rep{rep}"
        adata.obs['sample'] = sample_id
        adata.obs['day'] = day
        adata.obs['replicate'] = rep

        # Make cell barcodes unique across samples
        adata.obs_names = [f"{sample_id}_{bc}" for bc in adata.obs_names]

        print(f"  {sample_id}: {adata.n_obs} cells x {adata.n_vars} genes")
        adatas.append(adata)

    # Concatenate
    adata = ad.concat(adatas, join='outer')
    adata.obs_names_make_unique()

    # Ensure day is integer (for moscot time_key)
    adata.obs['day'] = adata.obs['day'].astype(int)

    print(f"\n  Merged: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Cells per day: {adata.obs['day'].value_counts().sort_index().to_dict()}")
    return adata


# ---------------------------------------------------------------------------
# Step 2: Quality control
# ---------------------------------------------------------------------------
def quality_control(adata):
    """Calculate QC metrics, detect doublets, and filter low-quality cells/genes."""
    print("\n" + "=" * 60)
    print("Step 2: Quality control")
    print("=" * 60)

    n_before = adata.n_obs

    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

    # Print pre-QC statistics
    print(f"\n  Pre-QC cells: {adata.n_obs}")
    print(f"  Median genes/cell: {np.median(adata.obs['n_genes_by_counts']):.0f}")
    print(f"  Median UMI/cell: {np.median(adata.obs['total_counts']):.0f}")
    print(f"  Median MT%: {np.median(adata.obs['pct_counts_mt']):.1f}%")

    # --- QC Plots ---
    # Temporarily make 'day' categorical for violin plots
    adata.obs['day_cat'] = adata.obs['day'].astype(str).astype('category')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    sc.pl.violin(adata, 'n_genes_by_counts', groupby='day_cat', ax=axes[0, 0], show=False)
    axes[0, 0].set_title('Genes per cell')
    axes[0, 0].set_ylabel('n_genes')

    sc.pl.violin(adata, 'total_counts', groupby='day_cat', ax=axes[0, 1], show=False)
    axes[0, 1].set_title('UMI counts per cell')
    axes[0, 1].set_ylabel('total_counts')

    sc.pl.violin(adata, 'pct_counts_mt', groupby='day_cat', ax=axes[0, 2], show=False)
    axes[0, 2].set_title('MT% per cell')
    axes[0, 2].set_ylabel('pct_counts_mt')

    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='pct_counts_mt',
                  ax=axes[1, 0], show=False)
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=axes[1, 1], show=False)

    sample_counts = adata.obs.groupby(['day', 'sample']).size()
    sample_counts.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_title('Cells per sample')
    axes[1, 2].set_ylabel('n_cells')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'qc_pre_filtering.png'), dpi=300, bbox_inches='tight')
    plt.close()
    del adata.obs['day_cat']
    print("  Saved: figures/qc_pre_filtering.png")

    # --- Filtering ---
    # Gene filter: expressed in at least 3 cells
    sc.pp.filter_genes(adata, min_cells=3)

    # Cell filters
    # Based on QC distribution analysis:
    #   - Median genes: 2312, 5th percentile: 1475
    #   - Median UMI: 6357, 5th percentile: 3334
    #   - MT%: 99th percentile = 8.3% (very clean data)
    sc.pp.filter_cells(adata, min_genes=200)
    adata = adata[adata.obs['n_genes_by_counts'] < 5000].copy()
    adata = adata[adata.obs['pct_counts_mt'] < 10].copy()

    print(f"\n  After basic QC: {adata.n_obs} cells")

    # --- Doublet detection (per sample) ---
    print("  Running doublet detection (per sample)...")
    adata.obs['doublet'] = False
    adata.obs['doublet_score'] = 0.0

    for sample in adata.obs['sample'].unique():
        mask = adata.obs['sample'] == sample
        sample_adata = adata[mask].copy()

        # DoubletDetection requires raw counts
        clf = doubletdetection.BoostClassifier(
            n_iters=10,
            clustering_algorithm='leiden',
            standard_scaling=True,
            random_state=42,
            verbose=False,
        )
        labels = clf.fit(sample_adata.X).predict()
        scores = clf.doublet_score()

        adata.obs.loc[mask, 'doublet'] = labels.astype(bool)
        adata.obs.loc[mask, 'doublet_score'] = scores

        n_doublets = labels.sum()
        print(f"    {sample}: {n_doublets}/{mask.sum()} doublets ({n_doublets/mask.sum()*100:.1f}%)")

    # Remove doublets
    n_doublets_total = adata.obs['doublet'].sum()
    adata = adata[~adata.obs['doublet']].copy()

    n_after = adata.n_obs
    print(f"\n  Post-QC + doublet removal: {n_after} cells (removed {n_before - n_after} total)")
    print(f"    Basic QC removed: {n_before - n_after - n_doublets_total}")
    print(f"    Doublets removed: {n_doublets_total}")
    print(f"  Cells per day: {adata.obs['day'].value_counts().sort_index().to_dict()}")
    print(f"  Target (paper): ~20,055 high-quality cells")

    return adata


# ---------------------------------------------------------------------------
# Step 3: Normalization + HVG selection
# ---------------------------------------------------------------------------
def normalize_and_select_hvg(adata):
    """Normalize, log-transform, select HVGs, regress, and scale."""
    print("\n" + "=" * 60)
    print("Step 3: Normalization and HVG selection")
    print("=" * 60)

    # Raw counts 보존 — moscot growth rate scoring (proliferation/apoptosis)에 필요
    adata.layers['counts'] = adata.X.copy()

    # Normalize to 10,000 counts per cell + log1p
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store normalized counts before HVG/scaling (for later use)
    adata.raw = adata.copy()

    # HVG selection (paper parameters)
    sc.pp.highly_variable_genes(
        adata,
        min_mean=0.01,
        max_mean=3,
        min_disp=0.5
    )
    n_hvg = adata.var['highly_variable'].sum()
    print(f"  HVGs selected: {n_hvg}")

    # regress_out 의도적으로 제거:
    # 원논문에서 regression 단계 명시 없음 + regress_out이 biological signal을
    # 왜곡하면 downstream OT cost에 영향. MT% < 10 필터로 충분.

    # Scale
    sc.pp.scale(adata, max_value=10)

    # HVG plot
    sc.pl.highly_variable_genes(adata, show=False)
    plt.savefig(os.path.join(FIGURE_DIR, 'hvg_selection.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/hvg_selection.png")

    return adata


# ---------------------------------------------------------------------------
# Step 4: Dimensionality reduction + clustering
# ---------------------------------------------------------------------------
def reduce_and_cluster(adata):
    """PCA, neighbors, UMAP, Leiden clustering."""
    print("\n" + "=" * 60)
    print("Step 4: Dimensionality reduction and clustering")
    print("=" * 60)

    # PCA (paper: 40 components)
    sc.tl.pca(adata, n_comps=40, svd_solver='arpack', random_state=42)
    print(f"  PCA: {adata.obsm['X_pca'].shape[1]} components")

    # Variance ratio plot
    sc.pl.pca_variance_ratio(adata, n_pcs=40, show=False)
    plt.savefig(os.path.join(FIGURE_DIR, 'pca_variance_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Neighbors (paper: 20 neighbors, 40 PCs)
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=40)
    print("  Neighbors: n_neighbors=20, n_pcs=40")

    # UMAP
    sc.tl.umap(adata, random_state=42)
    print("  UMAP computed")

    # Leiden clustering — 원논문은 Louvain이지만 Leiden이 modern standard.
    # 원논문의 11 cluster에 가장 가까운 resolution을 자동 선택
    TARGET_N_CLUSTERS = 11
    resolutions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    res_to_nclusters = {}

    for res in resolutions:
        key = f'leiden_{res}'
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=42)
        n_clusters = adata.obs[key].nunique()
        res_to_nclusters[res] = n_clusters
        print(f"  Leiden res={res}: {n_clusters} clusters")

    # Auto-select: closest to TARGET_N_CLUSTERS (tie-break: lower resolution for parsimony)
    best_res = min(resolutions, key=lambda r: (abs(res_to_nclusters[r] - TARGET_N_CLUSTERS), r))
    best_n = res_to_nclusters[best_res]
    print(f"\n  Auto-selected resolution: {best_res} ({best_n} clusters, target={TARGET_N_CLUSTERS})")

    sc.tl.leiden(adata, resolution=best_res, key_added='leiden', random_state=42)
    adata.uns['leiden_resolution'] = best_res

    return adata


# ---------------------------------------------------------------------------
# Step 5: Cell type annotation
# ---------------------------------------------------------------------------
def annotate_cell_types(adata):
    """Annotate cell types using marker genes from Ungricht et al."""
    print("\n" + "=" * 60)
    print("Step 5: Cell type annotation")
    print("=" * 60)

    # Filter marker genes to those present in the dataset
    available_markers = {}
    for ct, genes in MARKER_GENES.items():
        present = [g for g in genes if g in adata.raw.var_names]
        if present:
            available_markers[ct] = present
        else:
            print(f"  WARNING: No markers found for {ct}: {genes}")

    # Dotplot of marker genes
    sc.pl.dotplot(
        adata, available_markers, groupby='leiden',
        standard_scale='var', show=False
    )
    plt.savefig(os.path.join(FIGURE_DIR, 'marker_dotplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/marker_dotplot.png")

    # Heatmap of marker gene expression across clusters
    flat_markers = [g for genes in available_markers.values() for g in genes]
    flat_markers = list(dict.fromkeys(flat_markers))  # deduplicate, preserve order
    sc.pl.matrixplot(
        adata, flat_markers, groupby='leiden',
        standard_scale='var', cmap='viridis', show=False
    )
    plt.savefig(os.path.join(FIGURE_DIR, 'marker_matrixplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/marker_matrixplot.png")

    # --- Manual annotation ---
    # This mapping will need to be adjusted based on dotplot inspection.
    # Placeholder: we'll assign after examining the plots.
    # For now, create a helper function that can be called interactively.
    print("\n  [ACTION REQUIRED] Inspect marker_dotplot.png and marker_matrixplot.png")
    print("  Then define cluster_to_celltype mapping in assign_celltypes()")

    return adata


def assign_celltypes(adata, cluster_to_celltype):
    """
    Assign cell type labels based on manual inspection of marker plots.

    Parameters
    ----------
    cluster_to_celltype : dict
        Mapping from Leiden cluster ID (str) to cell type name.
        Example: {'0': 'eTEC', '1': 'PTEC', '2': 'Stroma_S1', ...}
    """
    adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_to_celltype)

    unmapped = adata.obs['cell_type'].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} cells have no cell type assignment")
        adata.obs['cell_type'] = adata.obs['cell_type'].fillna('Unknown')

    print(f"\n  Cell type distribution:")
    print(adata.obs['cell_type'].value_counts().to_string())
    return adata


# ---------------------------------------------------------------------------
# Step 6: Generate Fig 1 (UMAP overview)
# ---------------------------------------------------------------------------
def plot_fig1(adata):
    """Fig 1: UMAP colored by timepoint and cell type."""
    print("\n" + "=" * 60)
    print("Step 6: Generating Fig 1")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Panel A: UMAP by day
    sc.pl.umap(
        adata, color='day',
        palette={21: '#1f77b4', 28: '#ff7f0e', 35: '#2ca02c'},
        title='Collection timepoint',
        ax=axes[0], show=False
    )

    # Panel B: UMAP by cell type
    sc.pl.umap(
        adata, color='cell_type',
        title='Cell type annotation',
        ax=axes[1], show=False
    )

    # Panel C: UMAP by sample (batch check)
    sc.pl.umap(
        adata, color='sample',
        title='Sample / replicate',
        ax=axes[2], show=False
    )

    plt.suptitle('Kidney organoid scRNA-seq (Ungricht et al. 2022) — Preprocessing result',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig1_umap_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig1_umap_overview.png")

    # Supplementary: cell type proportion by day (stacked bar - compare with paper Fig 1C)
    ct_day = adata.obs.groupby(['day', 'cell_type']).size().unstack(fill_value=0)
    ct_day_frac = ct_day.div(ct_day.sum(axis=1), axis=0)
    ct_day_frac.plot(kind='bar', stacked=True, figsize=(8, 5))
    plt.title('Cell type proportions by timepoint')
    plt.ylabel('Fraction')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig1_supp_celltype_proportions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig1_supp_celltype_proportions.png")


# ---------------------------------------------------------------------------
# Step 7: Save processed data
# ---------------------------------------------------------------------------
def save_processed(adata):
    """Save preprocessed AnnData for downstream moscot analysis."""
    print("\n" + "=" * 60)
    print("Step 7: Saving processed data")
    print("=" * 60)

    out_path = os.path.join(PROCESSED_DIR, 'adata_preprocessed.h5ad')
    adata.write_h5ad(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Shape: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Layers: {list(adata.layers.keys())}")
    print(f"  Obs columns: {list(adata.obs.columns)}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("Ungricht et al. (2022) Kidney Organoid scRNA-seq Preprocessing")
    print("=" * 60)

    # Step 1: Load
    adata = load_and_merge_samples()

    # Step 2: QC
    adata = quality_control(adata)

    # Step 3: Normalize + HVG
    adata = normalize_and_select_hvg(adata)

    # Step 4: PCA + UMAP + Clustering
    adata = reduce_and_cluster(adata)

    # Step 5: Marker gene plots (annotation requires manual inspection)
    adata = annotate_cell_types(adata)

    # -----------------------------------------------------------------------
    # Cell type annotation based on marker gene expression analysis
    # Cross-referenced with Ungricht et al. Table S1A cluster definitions
    # Resolution auto-selected for closest to 11 clusters (paper).
    #
    # Annotation logic (matrixplot + dotplot inspection):
    # Leiden 0:  aS   — COL1A1+++, PDGFRB+, DCN+, ACTA2+. Activated stroma.
    # Leiden 1:  PTEC — CUBN+++, LRP2++, HNF4A+, EPCAM+, PAX8+. Proximal TEC.
    # Leiden 2:  P    — PODXL+++, NPHS1+, NPHS2+, EPCAM+. Mature podocyte.
    # Leiden 3:  S2   — COL1A1+, PDGFRB++, MKI67+, TOP2A+. Early stroma (cycling mix).
    # Leiden 4:  DTEC — EPCAM+, PAX8+, SLC12A1+, GATA3+, POU3F3+. Distal TEC.
    # Leiden 5:  eTEC — EPCAM+++, PAX8+++, CUBN+. Epithelial TEC.
    # Leiden 6:  S1   — COL1A1+, ACTA2++, TAGLN++. Transitional stroma.
    # Leiden 7:  LH   — COL1A1, DCN, ACTA2 moderate. Loop of Henle / minor stroma.
    # Leiden 8:  CC   — MKI67+++, TOP2A+++. Cycling cells (all timepoints).
    # Leiden 9:  eP   — EPCAM+, PAX8+, PODXL+, MAFB+. Early podocyte.
    #
    # Note: M (Muscle, off-target) is not a separate cluster at res=0.4.
    #   Original paper: M was a small off-target population (~3%).
    #   At this resolution it is absorbed into neighboring clusters.
    # -----------------------------------------------------------------------
    cluster_to_celltype = {
        '0': 'aS',
        '1': 'PTEC',
        '2': 'P',
        '3': 'S2',
        '4': 'DTEC',
        '5': 'eTEC',
        '6': 'S1',
        '7': 'LH',
        '8': 'CC',
        '9': 'eP',
    }
    adata = assign_celltypes(adata, cluster_to_celltype)
    plot_fig1(adata)

    # Save (even without cell type annotation, for checkpoint)
    save_processed(adata)

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("Next: Inspect marker plots → define cluster_to_celltype → run annotation")
    print("=" * 60)


if __name__ == '__main__':
    main()
