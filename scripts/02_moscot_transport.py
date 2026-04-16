"""
02_moscot_transport.py — OT 기반 kidney organoid trajectory 재구성 (Aim 1)

moscot TemporalProblem으로 Day 21/28/35 시점 간 cell-state transition을 확률적으로 추론.
핵심 산출물은 cell-type level transition matrix와 ancestor probability —
03_rsa_integration.py에서 perturbation vulnerability를 developmental transition에
매핑하는 좌표계로 사용됨.

Input:  data/processed/adata_preprocessed.h5ad
Output: data/processed/adata_with_transport.h5ad
        data/processed/transition_matrix_*.csv
        figures/fig2-4

OT 파라미터 선택 근거:
- epsilon=1e-3: sensitivity analysis로 검증 (Step 2b)
- tau_a/b=0.95: unbalanced OT — organoid에서 cell birth/death 허용
- scale_cost='mean': cost matrix normalization (moscot default)
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
FIGURE_DIR = os.path.join(PROJECT_DIR, 'figures')

os.makedirs(FIGURE_DIR, exist_ok=True)

# moscot parameters — Step 2b epsilon sensitivity에서 안정성 확인 완료
EPSILON = 1e-3       # entropic regularization
TAU_A = 0.95         # source marginal relaxation (unbalanced OT: birth/death 허용)
TAU_B = 0.95         # target marginal relaxation
SCALE_COST = 'mean'  # cost matrix normalization

# Epsilon sensitivity: 이 범위에서 transition matrix가 안정적이면 결과 신뢰 가능
EPSILON_CANDIDATES = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]


# ---------------------------------------------------------------------------
# Step 1: Load preprocessed data and prepare for moscot
# ---------------------------------------------------------------------------
def load_and_prepare():
    """Load preprocessed AnnData and verify it's ready for moscot."""
    print("=" * 60)
    print("Step 1: Loading preprocessed data")
    print("=" * 60)

    adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, 'adata_preprocessed.h5ad'))

    # Verify required fields
    assert 'day' in adata.obs.columns, "Missing 'day' column"
    assert 'cell_type' in adata.obs.columns, "Missing 'cell_type' column"
    assert 'X_pca' in adata.obsm, "Missing PCA embedding"
    assert 'X_umap' in adata.obsm, "Missing UMAP embedding"

    # moscot requires numeric time key
    adata.obs['day'] = adata.obs['day'].astype(int)

    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Timepoints: {sorted(adata.obs['day'].unique())}")
    print(f"  Cell types: {sorted(adata.obs['cell_type'].unique())}")
    print(f"  PCA dims: {adata.obsm['X_pca'].shape[1]}")

    return adata


# ---------------------------------------------------------------------------
# Step 2: Set up and solve TemporalProblem
# ---------------------------------------------------------------------------
def run_temporal_ot(adata):
    """Run moscot TemporalProblem: growth rate scoring, prepare, solve."""
    from moscot.problems.time import TemporalProblem

    print("\n" + "=" * 60)
    print("Step 2: moscot TemporalProblem")
    print("=" * 60)

    tp = TemporalProblem(adata)

    # Growth rate scoring — proliferation/apoptosis 기반으로 각 cell의
    # expected growth를 추정. unbalanced OT의 marginal prior로 사용됨
    print("\n  Scoring proliferation/apoptosis genes...")
    tp.score_genes_for_marginals(
        gene_set_proliferation='human',
        gene_set_apoptosis='human'
    )
    print(f"  Proliferation score range: "
          f"[{adata.obs['proliferation'].min():.2f}, {adata.obs['proliferation'].max():.2f}]")
    print(f"  Apoptosis score range: "
          f"[{adata.obs['apoptosis'].min():.2f}, {adata.obs['apoptosis'].max():.2f}]")

    # --- Prepare ---
    print(f"\n  Preparing transport problems (sequential policy)...")
    tp.prepare(
        time_key='day',
        joint_attr='X_pca',
        policy='sequential'     # Day21→28, Day28→35
    )
    print(f"  Transport problems: {list(tp.problems.keys())}")

    # --- Solve ---
    print(f"\n  Solving OT (epsilon={EPSILON}, tau_a={TAU_A}, tau_b={TAU_B})...")
    tp.solve(
        epsilon=EPSILON,
        tau_a=TAU_A,
        tau_b=TAU_B,
        scale_cost=SCALE_COST
    )
    print("  Solved.")

    return tp


# ---------------------------------------------------------------------------
# Step 2b: Epsilon sensitivity analysis
# ---------------------------------------------------------------------------
def epsilon_sensitivity_analysis(adata):
    """Test multiple epsilon values and compare transition matrices for stability."""
    from moscot.problems.time import TemporalProblem
    import seaborn as sns

    print("\n" + "=" * 60)
    print("Step 2b: Epsilon sensitivity analysis")
    print("=" * 60)

    results = {}

    for eps in EPSILON_CANDIDATES:
        print(f"\n  Solving with epsilon={eps}...")
        tp_test = TemporalProblem(adata)
        tp_test.score_genes_for_marginals(
            gene_set_proliferation='human',
            gene_set_apoptosis='human'
        )
        tp_test.prepare(time_key='day', joint_attr='X_pca', policy='sequential')
        tp_test.solve(
            epsilon=eps, tau_a=TAU_A, tau_b=TAU_B, scale_cost=SCALE_COST
        )

        # Compute Day 21→35 transition matrix for comparison
        t_mat = tp_test.cell_transition(
            source=21, target=35,
            source_groups='cell_type', target_groups='cell_type',
            forward=True, normalize=True
        )
        results[eps] = t_mat
        print(f"    Done. Matrix shape: {t_mat.shape}")

    # Compute pairwise Frobenius distance between transition matrices
    eps_list = sorted(results.keys())
    ref_eps = EPSILON  # our chosen epsilon
    print(f"\n  Frobenius distance from reference (epsilon={ref_eps}):")
    ref_mat = results[ref_eps]
    for eps in eps_list:
        dist = np.linalg.norm(results[eps].values - ref_mat.values, 'fro')
        print(f"    epsilon={eps}: dist={dist:.4f}")

    # Plot: heatmap grid of transition matrices across epsilons
    n_eps = len(eps_list)
    fig, axes = plt.subplots(1, n_eps, figsize=(6 * n_eps, 5))
    if n_eps == 1:
        axes = [axes]

    for i, eps in enumerate(eps_list):
        sns.heatmap(results[eps], cmap='YlOrRd', annot=True, fmt='.2f',
                    ax=axes[i], vmin=0, vmax=1, cbar=i == n_eps - 1,
                    xticklabels=True, yticklabels=(i == 0))
        marker = ' *' if eps == ref_eps else ''
        axes[i].set_title(f'ε={eps}{marker}', fontsize=11)
        if i == 0:
            axes[i].set_ylabel('Source cell type')
        axes[i].set_xlabel('Target cell type')

    plt.suptitle('Epsilon sensitivity: Day 21→35 transition matrix\n(* = selected)',
                 fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'supp_epsilon_sensitivity.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: supp_epsilon_sensitivity.png")

    return results


# ---------------------------------------------------------------------------
# Step 3: Compute cell-type transition matrices
# ---------------------------------------------------------------------------
def compute_transitions(tp, adata):
    """Compute cell-type level transition probability matrices."""
    print("\n" + "=" * 60)
    print("Step 3: Cell-type transition matrices")
    print("=" * 60)

    transitions = {}

    # Day 21 → Day 28
    print("\n  Computing Day 21 → Day 28 transitions...")
    t_21_28 = tp.cell_transition(
        source=21, target=28,
        source_groups='cell_type', target_groups='cell_type',
        forward=True, normalize=True,
        key_added='cell_transition_21_28'
    )
    transitions['21_28'] = t_21_28
    print(t_21_28.round(3).to_string())

    # Day 28 → Day 35
    print("\n  Computing Day 28 → Day 35 transitions...")
    t_28_35 = tp.cell_transition(
        source=28, target=35,
        source_groups='cell_type', target_groups='cell_type',
        forward=True, normalize=True,
        key_added='cell_transition_28_35'
    )
    transitions['28_35'] = t_28_35
    print(t_28_35.round(3).to_string())

    # Day 21 → Day 35 (full trajectory)
    print("\n  Computing Day 21 → Day 35 transitions (chained)...")
    t_21_35 = tp.cell_transition(
        source=21, target=35,
        source_groups='cell_type', target_groups='cell_type',
        forward=True, normalize=True,
        key_added='cell_transition_21_35'
    )
    transitions['21_35'] = t_21_35
    print(t_21_35.round(3).to_string())

    return transitions


# ---------------------------------------------------------------------------
# Step 4: Ancestor/Descendant analysis (pull/push)
# ---------------------------------------------------------------------------
def ancestor_descendant_analysis(tp, adata):
    """Compute ancestor (pull) and descendant (push) probabilities."""
    print("\n" + "=" * 60)
    print("Step 4: Ancestor/Descendant analysis")
    print("=" * 60)

    # Pull analysis: "Day 35의 X 세포가 Day 21에서 어디서 왔는가?"
    # eTEC (Day 21에 1% → Day 35에 43%)와 aS (0% → 49%)가
    # 주요 분석 대상 — 가장 극적인 population shift를 보이는 cell type들
    target_types = ['eTEC', 'DTEC', 'P', 'aS', 'S1', 'LH']

    for ct in target_types:
        # Check if this cell type exists at target timepoint
        mask = (adata.obs['day'] == 35) & (adata.obs['cell_type'] == ct)
        if mask.sum() == 0:
            print(f"  Skipping {ct}: no cells at Day 35")
            continue

        print(f"\n  Pull: ancestors of Day 35 {ct} (n={mask.sum()})...")
        try:
            tp.pull(
                source=21, target=35,
                data='cell_type', subset=ct,
                key_added=f'pull_{ct}',
                scale_by_marginals=True
            )
            print(f"    Saved to adata.obs['pull_{ct}']")
        except Exception as e:
            print(f"    Error: {e}")

    # Push analysis: "Day 21의 X가 Day 35에서 어디로 가는가?"
    # Day 21의 major population들 — S2(98%), PTEC(96%), eP(55%), DTEC(22%)
    source_types = ['PTEC', 'S2', 'eP', 'DTEC']

    for ct in source_types:
        mask = (adata.obs['day'] == 21) & (adata.obs['cell_type'] == ct)
        if mask.sum() == 0:
            print(f"  Skipping push {ct}: no cells at Day 21")
            continue

        print(f"\n  Push: descendants of Day 21 {ct} (n={mask.sum()})...")
        try:
            tp.push(
                source=21, target=35,
                data='cell_type', subset=ct,
                key_added=f'push_{ct}',
                scale_by_marginals=True
            )
            print(f"    Saved to adata.obs['push_{ct}']")
        except Exception as e:
            print(f"    Error: {e}")

    return adata


# ---------------------------------------------------------------------------
# Step 5: Mapping entropy (bifurcation point identification)
# ---------------------------------------------------------------------------
def compute_mapping_entropy(tp, adata):
    """Compute conditional entropy to identify cells with uncertain fate."""
    print("\n" + "=" * 60)
    print("Step 5: Mapping entropy")
    print("=" * 60)

    # 주의: full transport matrix materialization은 23K cells 기준 ~1.2TB → OOM.
    # batch_size로 메모리 분할 처리
    for src, tgt, key in [(21, 28, 'entropy_21_28'), (28, 35, 'entropy_28_35')]:
        print(f"  Computing entropy {src} → {tgt}...")
        try:
            tp.compute_entropy(source=src, target=tgt, key_added=key, batch_size=256)
        except Exception as e:
            print(f"    Skipped (memory/computation error): {type(e).__name__}")

    # Report high-entropy cells by cell type
    for period, key in [('21→28', 'entropy_21_28'), ('28→35', 'entropy_28_35')]:
        if key in adata.obs.columns:
            valid = adata.obs[key].dropna()
            print(f"\n  Entropy {period}: mean={valid.mean():.3f}, std={valid.std():.3f}")
            # Top entropy cell types (most uncertain fate)
            mean_by_ct = adata.obs.groupby('cell_type')[key].mean().dropna().sort_values(ascending=False)
            print(f"  Highest uncertainty:")
            for ct, val in mean_by_ct.head(5).items():
                print(f"    {ct}: {val:.3f}")

    return adata


# ---------------------------------------------------------------------------
# Fig 2: Ancestor probability on UMAP
# ---------------------------------------------------------------------------
def plot_fig2(adata):
    """Fig 2: Ancestor probability overlay on UMAP."""
    print("\n" + "=" * 60)
    print("Generating Fig 2: Ancestor probability")
    print("=" * 60)

    import moscot.plotting as mtp

    # Determine which pull keys exist
    pull_keys = [k for k in adata.obs.columns if k.startswith('pull_')]
    if not pull_keys:
        print("  No pull results found. Skipping Fig 2.")
        return

    n_panels = min(len(pull_keys), 4)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for i, key in enumerate(pull_keys[:n_panels]):
        ct_name = key.replace('pull_', '')
        try:
            mtp.pull(adata, key=key, basis='umap', ax=axes[i],
                     title=f'Ancestors of Day 35 {ct_name}')
        except Exception:
            # Fallback: plot on full UMAP, NaN cells shown in gray
            sc.pl.umap(adata, color=key, cmap='Reds', na_color='lightgray',
                       title=f'Ancestors of Day 35 {ct_name}',
                       ax=axes[i], show=False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig2_ancestor_probability.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig2_ancestor_probability.png")


# ---------------------------------------------------------------------------
# Fig 3: Transition matrix heatmap + Sankey
# ---------------------------------------------------------------------------
def plot_fig3(tp, adata, transitions):
    """Fig 3: Cell-type transition probability heatmap + Sankey diagram."""
    print("\n" + "=" * 60)
    print("Generating Fig 3: Transition matrix + Sankey")
    print("=" * 60)

    import seaborn as sns

    # --- Panel A: Heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    for i, (period, title) in enumerate([
        ('21_28', 'Day 21 → Day 28'),
        ('28_35', 'Day 28 → Day 35'),
        ('21_35', 'Day 21 → Day 35 (full)')
    ]):
        if period in transitions:
            df = transitions[period]
            sns.heatmap(df, cmap='YlOrRd', annot=True, fmt='.2f',
                        ax=axes[i], vmin=0, vmax=1,
                        annot_kws={'size': 11},
                        xticklabels=True, yticklabels=True,
                        cbar=i == 2,
                        cbar_kws={'shrink': 0.8, 'label': 'Transition probability'},
                        linewidths=0.5)
            axes[i].set_title(title, fontsize=13, fontweight='bold', pad=10)
            axes[i].set_xlabel('Target cell type', fontsize=11)
            axes[i].set_ylabel('Source cell type', fontsize=11)
            axes[i].tick_params(labelsize=10)

    plt.suptitle('Cell-type transition probability matrices (moscot OT)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig3_transition_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig3_transition_matrix.png")

    # --- Panel B: Sankey diagram ---
    import moscot.plotting as mtp
    try:
        tp.sankey(
            source=21, target=35,
            source_groups='cell_type', target_groups='cell_type',
            key_added='sankey_21_35'
        )
        fig, ax = plt.subplots(figsize=(14, 10))
        mtp.sankey(adata, key='sankey_21_35', title='Nephrogenic trajectory (Day 21 → 35)')
        plt.subplots_adjust(left=0.15, right=0.85)
        plt.savefig(os.path.join(FIGURE_DIR, 'fig3_sankey.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: fig3_sankey.png")
    except Exception as e:
        print(f"  Sankey diagram failed: {e}")
        print("  (Sankey is optional — heatmap is the primary figure)")


# ---------------------------------------------------------------------------
# Fig 4: OT vs Pseudotime comparison
# ---------------------------------------------------------------------------
def plot_fig4(adata):
    """Fig 4: DPT vs moscot comparison."""
    print("\n" + "=" * 60)
    print("Generating Fig 4: DPT vs moscot")
    print("=" * 60)

    # Compute Diffusion Pseudotime
    sc.tl.diffmap(adata, n_comps=15, random_state=42)

    # Root cell 선택: Day 21의 early progenitor 중 DC1 extremum.
    # 임의의 첫 번째 cell보다 낫지만, DPT 자체의 한계는 여전히 존재 —
    # 이 비교의 목적은 OT가 pseudotime 대비 어떤 정보를 추가하는지 보여주는 것
    early_mask = (adata.obs['day'] == 21) & (adata.obs['cell_type'].isin(['S2', 'PTEC', 'eP']))
    if early_mask.any():
        early_indices = np.flatnonzero(early_mask)
        dc1_vals = adata.obsm['X_diffmap'][early_indices, 0]
        # Pick the early cell with the lowest DC1 value (most undifferentiated end)
        adata.uns['iroot'] = early_indices[np.argmin(dc1_vals)]
        print(f"  Root cell: index {adata.uns['iroot']} "
              f"(DC1={dc1_vals.min():.4f}, cell_type={adata.obs['cell_type'].iloc[adata.uns['iroot']]})")
    else:
        adata.uns['iroot'] = np.argmin(adata.obsm['X_diffmap'][:, 0])
        print(f"  Root cell (fallback): index {adata.uns['iroot']}")

    sc.tl.dpt(adata)
    print(f"  DPT computed. Range: [{adata.obs['dpt_pseudotime'].min():.3f}, "
          f"{adata.obs['dpt_pseudotime'].max():.3f}]")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Panel A: DPT
    sc.pl.umap(adata, color='dpt_pseudotime', cmap='viridis',
               title='Diffusion Pseudotime', ax=axes[0], show=False)

    # Panel B: moscot ancestor probability (if available)
    pull_key = None
    for k in ['pull_PTEC', 'pull_aS', 'pull_DTEC', 'pull_P']:
        if k in adata.obs.columns:
            pull_key = k
            break

    if pull_key:
        ct_name = pull_key.replace('pull_', '')
        sc.pl.umap(adata, color=pull_key, cmap='Reds', na_color='lightgray',
                   title=f'moscot: {ct_name} ancestor probability',
                   ax=axes[1], show=False)
    else:
        axes[1].set_title('(No pull data available)')

    # Panel C: Mapping entropy
    entropy_key = None
    for k in ['entropy_21_28', 'entropy_28_35']:
        if k in adata.obs.columns:
            entropy_key = k
            break

    if entropy_key:
        sc.pl.umap(adata, color=entropy_key, cmap='YlOrRd', na_color='lightgray',
                   title=f'moscot: Mapping entropy ({entropy_key})',
                   ax=axes[2], show=False)
    else:
        axes[2].set_title('(No entropy data available)')

    plt.suptitle('Pseudotime vs Optimal Transport comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig4_dpt_vs_moscot.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig4_dpt_vs_moscot.png")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_results(adata, transitions):
    """Save annotated AnnData and transition matrices."""
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)

    # Save AnnData
    out_path = os.path.join(PROCESSED_DIR, 'adata_with_transport.h5ad')
    adata.write_h5ad(out_path)
    print(f"  Saved: {out_path}")

    # Save transition matrices as CSV
    for period, df in transitions.items():
        csv_path = os.path.join(PROCESSED_DIR, f'transition_matrix_{period}.csv')
        df.to_csv(csv_path)
        print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("moscot TemporalProblem: Kidney Organoid Trajectory Inference")
    print("=" * 60)

    # Step 1: Load
    adata = load_and_prepare()

    # Step 2: Run moscot
    tp = run_temporal_ot(adata)

    # Step 2b: Epsilon sensitivity analysis
    epsilon_sensitivity_analysis(adata)

    # Step 3: Transition matrices
    transitions = compute_transitions(tp, adata)

    # Step 4: Ancestor/Descendant
    adata = ancestor_descendant_analysis(tp, adata)

    # Step 5: Mapping entropy
    adata = compute_mapping_entropy(tp, adata)

    # Figures
    plot_fig2(adata)
    plot_fig3(tp, adata, transitions)
    plot_fig4(adata)

    # Save
    save_results(adata, transitions)

    print("\n" + "=" * 60)
    print("moscot analysis complete.")
    print("Next: Run 03_rsa_integration.py for Aim 2 (RSA score + transport map integration)")
    print("=" * 60)


if __name__ == '__main__':
    main()
