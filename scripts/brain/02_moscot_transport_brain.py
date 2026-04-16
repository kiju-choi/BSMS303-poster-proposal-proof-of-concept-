"""
02_moscot_transport_brain.py — OT 기반 brain organoid trajectory 재구성

Kidney pipeline과 동일한 moscot TemporalProblem을 brain organoid에 적용.
11 timepoints (10 sequential transitions)로 kidney보다 훨씬 촘촘한
trajectory 재구성이 가능 — generalizability + resolution 이점 확인.

Input:  data/fleck2022/processed/adata_preprocessed.h5ad
Output: data/fleck2022/processed/adata_with_transport.h5ad
        data/fleck2022/processed/transition_matrix_*.csv
        figures/brain/fig2-4

Cell type은 RG → IPC → EN/IN 분화 축이 핵심.
OT 파라미터는 kidney와 동일 (epsilon=1e-3, tau=0.95).
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
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'fleck2022', 'processed')
FIGURE_DIR = os.path.join(PROJECT_DIR, 'figures', 'brain')

os.makedirs(FIGURE_DIR, exist_ok=True)

RANDOM_STATE = 42

# moscot parameters
EPSILON = 1e-3
TAU_A = 0.95
TAU_B = 0.95
SCALE_COST = 'mean'

# Epsilon sensitivity
EPSILON_CANDIDATES = [1e-2, 5e-3, 1e-3, 5e-4]


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------
def load_and_prepare():
    """Load preprocessed brain organoid AnnData."""
    print("=" * 60)
    print("Step 1: Loading preprocessed brain organoid data")
    print("=" * 60)

    adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, 'adata_preprocessed.h5ad'))

    assert 'day' in adata.obs.columns
    assert 'cell_type' in adata.obs.columns
    assert 'X_pca' in adata.obsm

    adata.obs['day'] = adata.obs['day'].astype(int)

    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Timepoints: {sorted(adata.obs['day'].unique())}")
    print(f"  Cell types: {sorted(adata.obs['cell_type'].unique())}")

    return adata


# ---------------------------------------------------------------------------
# Step 2: Solve TemporalProblem
# ---------------------------------------------------------------------------
def run_temporal_ot(adata):
    """Run moscot TemporalProblem with sequential policy across 11 timepoints."""
    from moscot.problems.time import TemporalProblem

    print("\n" + "=" * 60)
    print("Step 2: moscot TemporalProblem (11 timepoints)")
    print("=" * 60)

    tp = TemporalProblem(adata)

    print("  Scoring proliferation/apoptosis genes...")
    tp.score_genes_for_marginals(
        gene_set_proliferation='human',
        gene_set_apoptosis='human'
    )

    print(f"  Preparing transport problems (sequential policy)...")
    tp.prepare(time_key='day', joint_attr='X_pca', policy='sequential')
    print(f"  Transport problems: {list(tp.problems.keys())}")

    print(f"  Solving OT (epsilon={EPSILON}, tau_a={TAU_A}, tau_b={TAU_B})...")
    tp.solve(epsilon=EPSILON, tau_a=TAU_A, tau_b=TAU_B, scale_cost=SCALE_COST)
    print("  Solved.")

    return tp


# ---------------------------------------------------------------------------
# Step 2b: Epsilon sensitivity
# ---------------------------------------------------------------------------
def epsilon_sensitivity_analysis(adata):
    """Epsilon stability 확인 — first→last transition matrix의 Frobenius distance로 비교."""
    from moscot.problems.time import TemporalProblem
    import seaborn as sns

    print("\n" + "=" * 60)
    print("Step 2b: Epsilon sensitivity analysis")
    print("=" * 60)

    days = sorted(adata.obs['day'].unique())
    first_day, last_day = days[0], days[-1]

    results = {}
    for eps in EPSILON_CANDIDATES:
        print(f"\n  Solving with epsilon={eps}...")
        tp_test = TemporalProblem(adata)
        tp_test.score_genes_for_marginals(
            gene_set_proliferation='human', gene_set_apoptosis='human')
        tp_test.prepare(time_key='day', joint_attr='X_pca', policy='sequential')
        tp_test.solve(epsilon=eps, tau_a=TAU_A, tau_b=TAU_B, scale_cost=SCALE_COST)

        t_mat = tp_test.cell_transition(
            source=first_day, target=last_day,
            source_groups='cell_type', target_groups='cell_type',
            forward=True, normalize=True
        )
        results[eps] = t_mat
        print(f"    Done. Matrix shape: {t_mat.shape}")

    # Frobenius distance from reference
    ref_mat = results[EPSILON]
    print(f"\n  Frobenius distance from reference (epsilon={EPSILON}):")
    for eps in sorted(results.keys()):
        dist = np.linalg.norm(results[eps].values - ref_mat.values, 'fro')
        print(f"    epsilon={eps}: dist={dist:.4f}")

    # Plot
    n_eps = len(EPSILON_CANDIDATES)
    fig, axes = plt.subplots(1, n_eps, figsize=(6 * n_eps, 5))
    if n_eps == 1:
        axes = [axes]

    for i, eps in enumerate(sorted(results.keys())):
        sns.heatmap(results[eps], cmap='YlOrRd', annot=True, fmt='.2f',
                    ax=axes[i], vmin=0, vmax=1, cbar=i == n_eps - 1,
                    xticklabels=True, yticklabels=(i == 0))
        marker = ' *' if eps == EPSILON else ''
        axes[i].set_title(f'ε={eps}{marker}', fontsize=11)
        if i == 0:
            axes[i].set_ylabel('Source cell type')
        axes[i].set_xlabel('Target cell type')

    plt.suptitle(f'Epsilon sensitivity: Day {first_day}→{last_day} transition matrix\n(* = selected)',
                 fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'supp_epsilon_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: supp_epsilon_sensitivity.png")

    return results


# ---------------------------------------------------------------------------
# Step 3: Transition matrices
# ---------------------------------------------------------------------------
def compute_transitions(tp, adata):
    """Compute cell-type transition matrices for key intervals."""
    print("\n" + "=" * 60)
    print("Step 3: Cell-type transition matrices")
    print("=" * 60)

    days = sorted(adata.obs['day'].unique())
    transitions = {}

    # Adjacent transitions
    for i in range(len(days) - 1):
        src, tgt = int(days[i]), int(days[i + 1])
        key = f'{src}_{tgt}'
        print(f"\n  Computing Day {src} → Day {tgt}...")
        t_mat = tp.cell_transition(
            source=src, target=tgt,
            source_groups='cell_type', target_groups='cell_type',
            forward=True, normalize=True
        )
        transitions[key] = t_mat
        print(t_mat.round(3).to_string())

    # Full trajectory (first → last)
    first_day, last_day = int(days[0]), int(days[-1])
    key_full = f'{first_day}_{last_day}'
    print(f"\n  Computing Day {first_day} → Day {last_day} (full trajectory)...")
    t_full = tp.cell_transition(
        source=first_day, target=last_day,
        source_groups='cell_type', target_groups='cell_type',
        forward=True, normalize=True
    )
    transitions[key_full] = t_full
    print(t_full.round(3).to_string())

    return transitions


# ---------------------------------------------------------------------------
# Step 4: Ancestor/Descendant analysis
# ---------------------------------------------------------------------------
def ancestor_descendant_analysis(tp, adata):
    """Pull/push analysis for key neuronal cell types."""
    print("\n" + "=" * 60)
    print("Step 4: Ancestor/Descendant analysis")
    print("=" * 60)

    days = sorted(adata.obs['day'].unique())
    first_day, last_day = int(days[0]), int(days[-1])

    # Pull: ancestors of late cell types
    target_types = ['EN', 'IN', 'IPC', 'RG']
    for ct in target_types:
        mask = (adata.obs['day'] == last_day) & (adata.obs['cell_type'] == ct)
        if mask.sum() == 0:
            print(f"  Skipping pull {ct}: no cells at Day {last_day}")
            continue

        print(f"\n  Pull: ancestors of Day {last_day} {ct} (n={mask.sum()})...")
        try:
            tp.pull(source=first_day, target=last_day,
                    data='cell_type', subset=ct,
                    key_added=f'pull_{ct}', scale_by_marginals=True)
            print(f"    Saved to adata.obs['pull_{ct}']")
        except Exception as e:
            print(f"    Error: {e}")

    # Push: descendants of early progenitors
    source_types = ['RG', 'IPC']
    for ct in source_types:
        mask = (adata.obs['day'] == first_day) & (adata.obs['cell_type'] == ct)
        if mask.sum() == 0:
            print(f"  Skipping push {ct}: no cells at Day {first_day}")
            continue

        print(f"\n  Push: descendants of Day {first_day} {ct} (n={mask.sum()})...")
        try:
            tp.push(source=first_day, target=last_day,
                    data='cell_type', subset=ct,
                    key_added=f'push_{ct}', scale_by_marginals=True)
            print(f"    Saved to adata.obs['push_{ct}']")
        except Exception as e:
            print(f"    Error: {e}")

    return adata


# ---------------------------------------------------------------------------
# Fig 2: Ancestor probability
# ---------------------------------------------------------------------------
def plot_fig2(adata):
    """Ancestor probability UMAP overlay.
    Percentile-based vmax clipping 사용 — probability mass가 소수 cell에 집중되면
    gradient가 안 보이는 문제 방지.
    """
    from matplotlib.colors import LinearSegmentedColormap

    print("\n" + "=" * 60)
    print("Generating Fig 2: Ancestor probability (brain)")
    print("=" * 60)

    pull_keys = [k for k in adata.obs.columns if k.startswith('pull_')]
    if not pull_keys:
        print("  No pull results. Skipping.")
        return

    n_panels = min(len(pull_keys), 4)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    days = sorted(adata.obs['day'].unique())
    last_day = int(days[-1])

    # Custom colormap: light gray → orange → dark red (more visible than plain Reds)
    cmap_ancestor = LinearSegmentedColormap.from_list(
        'ancestor', ['#f0f0f0', '#fee08b', '#fc8d59', '#d73027', '#7f0000'])

    for i, key in enumerate(pull_keys[:n_panels]):
        ct_name = key.replace('pull_', '')
        vals = adata.obs[key].values
        nonzero = vals[vals > 0]

        # Percentile-based vmax: clip at 99th percentile of non-zero values
        if len(nonzero) > 0:
            vmax = np.percentile(nonzero, 99)
            vmax = max(vmax, np.percentile(nonzero, 95))  # safety floor
        else:
            vmax = None

        print(f"  {key}: non-zero={len(nonzero)}, "
              f"median={np.median(nonzero) if len(nonzero) else 0:.4g}, "
              f"99pct={vmax}")

        sc.pl.umap(adata, color=key, cmap=cmap_ancestor,
                   na_color='#e8e8e8', vmin=0, vmax=vmax,
                   title=f'Ancestors of Day {last_day} {ct_name}',
                   ax=axes[i], show=False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig2_ancestor_probability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig2_ancestor_probability.png")


# ---------------------------------------------------------------------------
# Fig 3: Transition heatmaps (selected intervals)
# ---------------------------------------------------------------------------
def plot_fig3(transitions):
    """Transition probability heatmap — early/mid/late adjacent + full trajectory.
    10개 adjacent matrix를 전부 그리면 가독성이 떨어지므로,
    대표적인 3개 + full trajectory 1개만 선택.
    """
    print("\n" + "=" * 60)
    print("Generating Fig 3: Transition matrices (brain)")
    print("=" * 60)

    import seaborn as sns

    all_keys = sorted(transitions.keys(), key=lambda k: int(k.split('_')[0]))

    # Separate adjacent (consecutive timepoint) from full-trajectory keys.
    # Adjacent keys: the day span equals the gap between two consecutive
    # timepoints.  The full-trajectory key has the largest span overall.
    spans = {k: int(k.split('_')[1]) - int(k.split('_')[0]) for k in all_keys}
    max_span_key = max(spans, key=spans.get)
    adjacent = [k for k in all_keys if k != max_span_key]

    # Pick early / mid / late from the adjacent list
    display_keys = []
    if len(adjacent) >= 3:
        display_keys = [adjacent[0], adjacent[len(adjacent) // 2], adjacent[-1]]
    elif adjacent:
        display_keys = list(adjacent)
    display_keys.append(max_span_key)

    labels = {max_span_key: 'full trajectory'}
    for idx, k in enumerate(display_keys):
        if k == max_span_key:
            continue
        if idx == 0:
            labels[k] = 'early'
        elif idx == len(display_keys) - 2:
            labels[k] = 'late'
        else:
            labels[k] = 'mid'

    print(f"  Selected panels: {display_keys}")

    n = len(display_keys)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6))
    if n == 1:
        axes = [axes]

    for i, key in enumerate(display_keys):
        src, tgt = key.split('_')
        phase = labels.get(key, '')
        title = f'Day {src} → Day {tgt}'
        if phase:
            title += f'  ({phase})'

        sns.heatmap(transitions[key], cmap='YlOrRd', annot=True, fmt='.2f',
                    ax=axes[i], vmin=0, vmax=1,
                    annot_kws={'size': 11},
                    xticklabels=True, yticklabels=True,
                    cbar=i == n - 1,
                    cbar_kws={'shrink': 0.8, 'label': 'Transition probability'})
        axes[i].set_title(title, fontsize=13, fontweight='bold', pad=10)
        axes[i].set_xlabel('Target cell type', fontsize=11)
        axes[i].set_ylabel('Source cell type', fontsize=11)
        axes[i].tick_params(labelsize=10)

    plt.suptitle('Cell-type transition probability matrices (moscot OT) — Brain organoid',
                 fontsize=15, y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig3_transition_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig3_transition_matrix.png")


# ---------------------------------------------------------------------------
# Fig 4: DPT vs moscot
# ---------------------------------------------------------------------------
def plot_fig4(adata):
    """Fig 4: DPT vs moscot comparison."""
    print("\n" + "=" * 60)
    print("Generating Fig 4: DPT vs moscot (brain)")
    print("=" * 60)

    sc.tl.diffmap(adata, n_comps=15, random_state=RANDOM_STATE)

    # Root: 가장 이른 시점의 RG 중 DC1 extremum — DPT 한계는 있지만 OT와의 비교용
    day_int = adata.obs['day'].astype(int)
    early_mask = (day_int == day_int.min()) & \
                 (adata.obs['cell_type'] == 'RG')
    if early_mask.any():
        early_indices = np.flatnonzero(early_mask)
        dc1_vals = adata.obsm['X_diffmap'][early_indices, 0]
        adata.uns['iroot'] = early_indices[np.argmin(dc1_vals)]
        print(f"  Root cell: index {adata.uns['iroot']} "
              f"(DC1={dc1_vals.min():.4f}, cell_type=RG)")
    else:
        adata.uns['iroot'] = np.argmin(adata.obsm['X_diffmap'][:, 0])

    sc.tl.dpt(adata)
    print(f"  DPT computed. Range: [{adata.obs['dpt_pseudotime'].min():.3f}, "
          f"{adata.obs['dpt_pseudotime'].max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    sc.pl.umap(adata, color='dpt_pseudotime', cmap='viridis',
               title='Diffusion Pseudotime', ax=axes[0], show=False)

    pull_key = None
    for k in ['pull_EN', 'pull_IN', 'pull_IPC']:
        if k in adata.obs.columns:
            pull_key = k
            break

    if pull_key:
        ct_name = pull_key.replace('pull_', '')
        sc.pl.umap(adata, color=pull_key, cmap='Reds', na_color='lightgray',
                   title=f'moscot: {ct_name} ancestor probability',
                   ax=axes[1], show=False)
    else:
        axes[1].set_title('(No pull data)')

    sc.pl.umap(adata, color='day', cmap='viridis',
               title='Collection timepoint (day)', ax=axes[2], show=False)

    plt.suptitle('Pseudotime vs Optimal Transport — Brain organoid', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig4_dpt_vs_moscot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig4_dpt_vs_moscot.png")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_results(adata, transitions):
    """Save results."""
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)

    out_path = os.path.join(PROCESSED_DIR, 'adata_with_transport.h5ad')
    adata.write_h5ad(out_path)
    print(f"  Saved: {out_path}")

    for period, df in transitions.items():
        csv_path = os.path.join(PROCESSED_DIR, f'transition_matrix_{period}.csv')
        df.to_csv(csv_path)
        print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("moscot TemporalProblem: Brain Organoid Trajectory Inference")
    print("=" * 60)

    adata = load_and_prepare()
    tp = run_temporal_ot(adata)
    epsilon_sensitivity_analysis(adata)
    transitions = compute_transitions(tp, adata)
    adata = ancestor_descendant_analysis(tp, adata)

    plot_fig2(adata)
    plot_fig3(transitions)
    plot_fig4(adata)
    save_results(adata, transitions)

    print("\n" + "=" * 60)
    print("Brain organoid moscot analysis complete.")
    print("Next: Run 03_perturbation_integration_brain.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
