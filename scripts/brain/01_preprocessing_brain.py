"""
01_preprocessing_brain.py — Brain organoid scRNA-seq 전처리

Fleck et al. (2022) brain organoid 데이터 전처리.
Kidney pipeline과 동일한 구조를 따르되, 데이터 특성에 맞게 조정.
두 번째 organ system에서 같은 파이프라인이 작동하는지 —
generalizability validation이 핵심 목적.

Input:  data/fleck2022/raw/RNA_data.h5ad (Zenodo, 49,718 cells, 11 timepoints)
Output: data/fleck2022/processed/adata_preprocessed.h5ad

Kidney와의 주요 차이:
- 11 timepoints (day 4-61) → time resolution이 훨씬 촘촘
- Seurat SCTransform으로 이미 log-normalized → raw counts 없음
- Cell type: Nowakowski reference atlas prediction 사용
- 4 cell lines — batch effect 가능성 있으나 UMAP에서 확인
"""

import scanpy as sc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_DIR, 'data', 'fleck2022', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'fleck2022', 'processed')
FIGURE_DIR = os.path.join(PROJECT_DIR, 'figures', 'brain')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42

# Target cluster count (Fleck et al. identified ~7 major lineages)
TARGET_N_CLUSTERS = 7


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_data():
    """Load Fleck et al. h5ad (already log-normalized by Seurat)."""
    print("=" * 60)
    print("Step 1: Loading Fleck et al. brain organoid data")
    print("=" * 60)

    adata = sc.read_h5ad(os.path.join(RAW_DIR, 'RNA_data.h5ad'))

    # Rename 'age' to 'day' for consistency with kidney pipeline
    adata.obs['day'] = adata.obs['age'].astype(int)

    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Timepoints: {sorted(adata.obs['day'].unique())}")
    print(f"  Cell lines: {adata.obs['line'].value_counts().to_dict()}")
    print(f"  Cells per day:")
    print(adata.obs['day'].value_counts().sort_index().to_string())

    return adata


# ---------------------------------------------------------------------------
# Step 2: Quality control
# ---------------------------------------------------------------------------
def quality_control(adata):
    """Filter low-quality cells. Data is already Seurat-filtered but apply basic QC."""
    print("\n" + "=" * 60)
    print("Step 2: Quality control")
    print("=" * 60)

    n_before = adata.n_obs

    # QC metrics (mito already computed as 'percent_mito')
    print(f"  Pre-QC cells: {adata.n_obs}")
    print(f"  Median genes/cell: {np.median(adata.obs['nFeature_RNA']):.0f}")
    print(f"  Median UMI/cell: {np.median(adata.obs['nCount_RNA']):.0f}")
    print(f"  Median MT%: {np.median(adata.obs['percent_mito'])*100:.1f}%")

    # QC plots
    adata.obs['day_cat'] = adata.obs['day'].astype(str).astype('category')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sc.pl.violin(adata, 'nFeature_RNA', groupby='day_cat', ax=axes[0], show=False)
    axes[0].set_title('Genes per cell')
    sc.pl.violin(adata, 'nCount_RNA', groupby='day_cat', ax=axes[1], show=False)
    axes[1].set_title('UMI counts per cell')
    sc.pl.violin(adata, 'percent_mito', groupby='day_cat', ax=axes[2], show=False)
    axes[2].set_title('MT fraction per cell')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'qc_pre_filtering.png'), dpi=300, bbox_inches='tight')
    plt.close()
    del adata.obs['day_cat']
    print("  Saved: figures/brain/qc_pre_filtering.png")

    # Filter: build combined mask first, then subset once (memory-efficient)
    keep_mask = np.ones(adata.n_obs, dtype=bool)

    if 'p_singlet' in adata.obs.columns:
        doublet_mask = adata.obs['p_singlet'] > 0.5
        n_doublets = (~doublet_mask).sum()
        keep_mask &= doublet_mask.values
        print(f"  Doublet filter (p_singlet > 0.5): removing {n_doublets}")

    mito_mask = adata.obs['percent_mito'] < 0.1
    n_mito = (~mito_mask).sum()
    keep_mask &= mito_mask.values
    print(f"  MT filter (< 10%): removing {n_mito}")

    # Single subset + copy
    adata = adata[keep_mask].copy()

    n_after = adata.n_obs
    print(f"  Post-QC: {n_after} cells (removed {n_before - n_after})")
    print(f"  Cells per day: {adata.obs['day'].value_counts().sort_index().to_dict()}")

    return adata


# ---------------------------------------------------------------------------
# Step 3: HVG selection + scaling (data already log-normalized)
# ---------------------------------------------------------------------------
def select_hvg_and_scale(adata):
    """Select HVGs and scale. Data is already log-normalized from Seurat."""
    print("\n" + "=" * 60)
    print("Step 3: HVG selection and scaling")
    print("=" * 60)

    # Store current expression as 'raw' for later marker gene plotting
    adata.raw = adata.copy()

    # HVG selection — seurat flavor 사용 (seurat_v3는 raw counts 필요, 여기엔 없음)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    n_hvg = adata.var['highly_variable'].sum()
    print(f"  HVGs selected: {n_hvg}")

    # HVG subset 먼저 — 44K x 33K full matrix를 scale하면 메모리 폭발
    adata = adata[:, adata.var['highly_variable']].copy()
    print(f"  Subset to HVGs: {adata.n_vars} genes")

    # Scale — regress_out 제거 (kidney pipeline과 동일 근거)
    sc.pp.scale(adata, max_value=10)

    # HVG plot
    sc.pl.highly_variable_genes(adata, show=False)
    plt.savefig(os.path.join(FIGURE_DIR, 'hvg_selection.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/brain/hvg_selection.png")

    return adata


# ---------------------------------------------------------------------------
# Step 4: Dimensionality reduction + clustering
# ---------------------------------------------------------------------------
def reduce_and_cluster(adata):
    """PCA, neighbors, UMAP, Leiden clustering with auto-resolution."""
    print("\n" + "=" * 60)
    print("Step 4: Dimensionality reduction and clustering")
    print("=" * 60)

    # PCA
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack', random_state=RANDOM_STATE)
    print(f"  PCA: {adata.obsm['X_pca'].shape[1]} components")

    sc.pl.pca_variance_ratio(adata, n_pcs=50, show=False)
    plt.savefig(os.path.join(FIGURE_DIR, 'pca_variance_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Neighbors
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=30, random_state=RANDOM_STATE)
    print("  Neighbors: n_neighbors=20, n_pcs=30")

    # UMAP
    sc.tl.umap(adata, random_state=RANDOM_STATE)
    print("  UMAP computed")

    # Leiden auto-resolution
    resolutions = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
    res_to_nclusters = {}

    for res in resolutions:
        key = f'leiden_{res}'
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=RANDOM_STATE)
        n_clusters = adata.obs[key].nunique()
        res_to_nclusters[res] = n_clusters
        print(f"  Leiden res={res}: {n_clusters} clusters")

    best_res = min(resolutions, key=lambda r: (abs(res_to_nclusters[r] - TARGET_N_CLUSTERS), r))
    best_n = res_to_nclusters[best_res]
    print(f"\n  Auto-selected resolution: {best_res} ({best_n} clusters, target={TARGET_N_CLUSTERS})")

    sc.tl.leiden(adata, resolution=best_res, key_added='leiden', random_state=RANDOM_STATE)
    adata.uns['leiden_resolution'] = best_res

    return adata


# ---------------------------------------------------------------------------
# Step 5: Cell type annotation
# ---------------------------------------------------------------------------
def annotate_cell_types(adata):
    """Use Nowakowski prediction as primary annotation, refine with markers."""
    print("\n" + "=" * 60)
    print("Step 5: Cell type annotation")
    print("=" * 60)

    # Nowakowski atlas prediction → broad category로 매핑.
    # Trajectory 분석에는 fine-grained subtype보다 lineage-level이 적합
    celltype_map = {
        'RG': 'RG',           # Radial glia
        'IPC': 'IPC',         # Intermediate progenitor cells
        'EN': 'EN',           # Excitatory neurons
        'IN': 'IN',           # Interneurons
        'Choroid': 'Choroid',
        'Astrocyte': 'Astro',
        'OPC': 'OPC',
        'Mural': 'Other',
        'Others': 'Other',
        'U1': 'Other',
        'U2': 'Other',
        'U3': 'Other',
    }

    adata.obs['cell_type'] = adata.obs['nowakowski_prediction'].map(celltype_map)
    unmapped = adata.obs['cell_type'].isna().sum()
    if unmapped > 0:
        adata.obs['cell_type'] = adata.obs['cell_type'].fillna('Other')
        print(f"  WARNING: {unmapped} unmapped cells → 'Other'")

    print("\n  Cell type distribution:")
    print(adata.obs['cell_type'].value_counts().to_string())

    # Cross-tabulate cell type by timepoint
    ct_by_day = adata.obs.groupby(['day', 'cell_type']).size().unstack(fill_value=0)
    print("\n  Cell types by timepoint:")
    print(ct_by_day.to_string())

    # Marker genes for brain organoid cell types
    brain_markers = {
        'RG': ['VIM', 'HES1', 'SOX2', 'PAX6', 'FABP7'],
        'IPC': ['EOMES', 'NEUROG2', 'NEUROD1'],
        'EN': ['NEUROD6', 'SLC17A7', 'TBR1', 'BCL11B'],
        'IN': ['DLX2', 'DLX5', 'GAD1', 'GAD2'],
    }

    # Filter to available markers
    available_markers = {}
    for ct, genes in brain_markers.items():
        present = [g for g in genes if g in adata.raw.var_names]
        if present:
            available_markers[ct] = present

    if available_markers:
        sc.pl.dotplot(adata, available_markers, groupby='leiden',
                      standard_scale='var', show=False)
        plt.savefig(os.path.join(FIGURE_DIR, 'marker_dotplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/brain/marker_dotplot.png")

    return adata


# ---------------------------------------------------------------------------
# Step 6: Fig 1 — UMAP overview
# ---------------------------------------------------------------------------
def plot_fig1(adata):
    """Fig 1: UMAP colored by timepoint, cell type, and cell line."""
    print("\n" + "=" * 60)
    print("Step 6: Generating Fig 1 (brain)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    sc.pl.umap(adata, color='day', cmap='viridis',
               title='Developmental timepoint (day)', ax=axes[0], show=False)
    sc.pl.umap(adata, color='cell_type',
               title='Cell type (Nowakowski)', ax=axes[1], show=False)
    sc.pl.umap(adata, color='line',
               title='Cell line', ax=axes[2], show=False)

    plt.suptitle('Brain organoid scRNA-seq (Fleck et al. 2022) — Preprocessing result',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig1_umap_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/brain/fig1_umap_overview.png")

    # Cell type proportion by day
    ct_day = adata.obs.groupby(['day', 'cell_type']).size().unstack(fill_value=0)
    ct_day_frac = ct_day.div(ct_day.sum(axis=1), axis=0)
    ct_day_frac.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.title('Cell type proportions by developmental timepoint')
    plt.ylabel('Fraction')
    plt.xlabel('Day')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig1_supp_celltype_proportions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/brain/fig1_supp_celltype_proportions.png")


# ---------------------------------------------------------------------------
# Step 7: Save
# ---------------------------------------------------------------------------
def save_processed(adata):
    """Save preprocessed AnnData."""
    print("\n" + "=" * 60)
    print("Step 7: Saving processed data")
    print("=" * 60)

    out_path = os.path.join(PROCESSED_DIR, 'adata_preprocessed.h5ad')
    adata.write_h5ad(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Shape: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Obs columns: {list(adata.obs.columns)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Fleck et al. (2022) Brain Organoid scRNA-seq Preprocessing")
    print("=" * 60)

    adata = load_data()
    adata = quality_control(adata)
    adata = select_hvg_and_scale(adata)
    adata = reduce_and_cluster(adata)
    adata = annotate_cell_types(adata)
    plot_fig1(adata)
    save_processed(adata)

    print("\n" + "=" * 60)
    print("Brain organoid preprocessing complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
