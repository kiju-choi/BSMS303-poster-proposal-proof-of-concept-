"""
03_perturbation_integration_brain.py — Perturbation과 trajectory 통합 (Aim 2, Brain)

Kidney의 03_rsa_integration.py와 동일한 로직이지만,
perturbation 데이터가 CRISPR screening(RSA) → CRISPRi(log_odds_ratio)로 다름.

Fleck et al. CRISPRi: 20 TFs x 3 lineages
- ctx (cortical/dorsal) → Excitatory neuron trajectory
- ge (ganglionic eminence/ventral) → Interneuron trajectory
- nt (neural tube) → Progenitor maintenance

VS = OT_weight_norm x Sum(|log_odds_ratio| x relevance_weight)

Input:  data/fleck2022/raw/ST5_CRISPRi_enrichment.xls
        data/fleck2022/processed/transition_matrix_*.csv
Output: figures/brain/fig5-7, vulnerability_scores.csv

통계적 보강은 kidney와 동일: OT normalization, permutation test, naive 비교.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_DIR, 'data', 'fleck2022', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'fleck2022', 'processed')
FIGURE_DIR = os.path.join(PROJECT_DIR, 'figures', 'brain')

os.makedirs(FIGURE_DIR, exist_ok=True)

EXCEL_PATH = os.path.join(RAW_DIR, 'ST5_CRISPRi_enrichment.xls')

# Relevance weights — kidney의 disease category weight 대신 effect size 기반.
# Brain TF들은 모두 high-confidence regulator이므로 LOR 크기로만 구분
RELEVANCE_WEIGHTS = {
    'strong_effect': 1.0,    # |LOR| > 1 and p < 0.01
    'moderate_effect': 0.7,  # |LOR| > 0.5
    'weak_effect': 0.3,      # 나머지
}

N_PERMUTATIONS = 10000


# ---------------------------------------------------------------------------
# Step 1: Load perturbation data
# ---------------------------------------------------------------------------
def load_perturbation_data():
    """Load CRISPRi enrichment from Supplementary Table 5."""
    print("=" * 60)
    print("Step 1: Loading CRISPRi perturbation data (ST5)")
    print("=" * 60)

    df = pd.read_excel(EXCEL_PATH, sheet_name=0)

    # Extract gene name from guide name (e.g., 'GLI3-1' -> 'GLI3')
    df['gene'] = df['x'].str.replace(r'-\d+$', '', regex=True)

    print(f"  Total rows: {len(df)}")
    print(f"  TFs: {sorted(df['gene'].unique())}")
    print(f"  Lineages: {df['y'].unique()}")
    print(f"  Groups: {df['group'].unique()}")

    # Use CMH test results (group=NaN) as the primary gene-level summary
    cmh = df[df['group'].isna()].copy()
    print(f"\n  CMH test rows: {len(cmh)}")

    # Aggregate per gene (average across guides for the same gene+lineage)
    gene_summary = cmh.groupby(['gene', 'y']).agg(
        log_odds_ratio=('log_odds_ratio', 'mean'),
        pval=('pval', lambda x: x.min()),  # most significant
        n_guides=('x', 'count'),
    ).reset_index()

    print(f"  Gene-level summaries: {len(gene_summary)}")
    print(f"\n  Per-lineage effect summary:")
    for lineage in ['ctx', 'ge', 'nt']:
        sub = gene_summary[gene_summary['y'] == lineage]
        n_depleted = (sub['log_odds_ratio'] < -0.5).sum()
        n_enriched = (sub['log_odds_ratio'] > 0.5).sum()
        print(f"    {lineage}: {n_depleted} depleted, {n_enriched} enriched (of {len(sub)} TFs)")

    return df, gene_summary


# ---------------------------------------------------------------------------
# Step 2: Classify perturbation patterns
# ---------------------------------------------------------------------------
def classify_patterns(gene_summary):
    """Classify each TF by its enrichment/depletion pattern across lineages."""
    print("\n" + "=" * 60)
    print("Step 2: Classifying TF perturbation patterns")
    print("=" * 60)

    patterns = {}
    for gene in gene_summary['gene'].unique():
        sub = gene_summary[gene_summary['gene'] == gene]
        gene_pats = []
        for _, row in sub.iterrows():
            lineage = row['y']
            lor = row['log_odds_ratio']
            pval = row['pval']

            if lor < -0.5 and pval < 0.01:
                gene_pats.append(f'{lineage}_DOWN')
            elif lor > 0.5 and pval < 0.01:
                gene_pats.append(f'{lineage}_UP')

        if gene_pats:
            patterns[gene] = gene_pats

    print("  TF patterns:")
    for gene, pats in sorted(patterns.items()):
        print(f"    {gene}: {', '.join(pats)}")

    return patterns


# ---------------------------------------------------------------------------
# Step 3: OT-weighted transition attribution
# ---------------------------------------------------------------------------
def load_adjacent_transition_matrices():
    """Load all adjacent (sequential) transition matrices."""
    tm_files = [f for f in os.listdir(PROCESSED_DIR) if f.startswith('transition_matrix_')]
    matrices = {}
    spans = {}
    for f in tm_files:
        parts = f.replace('transition_matrix_', '').replace('.csv', '').split('_')
        if len(parts) == 2:
            span = int(parts[1]) - int(parts[0])
            spans[f] = span
            matrices[f] = pd.read_csv(os.path.join(PROCESSED_DIR, f), index_col=0)

    # Separate adjacent from full-trajectory (largest span)
    if not spans:
        return {}, None
    max_span_file = max(spans, key=spans.get)
    adjacent = {k: v for k, v in matrices.items() if k != max_span_file}
    return adjacent, matrices[max_span_file]


def compute_ot_weights_from_adjacent(adjacent_matrices, cell_type_pairs):
    """Compute mean OT weight for cell type pairs across all adjacent matrices.

    For each adjacent matrix, look up each (src, tgt) pair. Average across
    all matrices where both src and tgt exist. This captures the transition
    probability at every developmental stage, not just the Day 4 starting point.
    """
    pair_probs = {pair: [] for pair in cell_type_pairs}

    for fname, tm in adjacent_matrices.items():
        for src, tgt in cell_type_pairs:
            if src in tm.index and tgt in tm.columns:
                pair_probs[(src, tgt)].append(tm.loc[src, tgt])

    total_weight = 0.0
    pair_details = []
    for (src, tgt), probs in pair_probs.items():
        if probs:
            mean_prob = np.mean(probs)
            total_weight += mean_prob
            pair_details.append(f"{src}->{tgt}={mean_prob:.3f} (n={len(probs)})")
        else:
            pair_details.append(f"{src}->{tgt}=N/A")

    return total_weight, pair_details


def attribute_to_transitions(patterns, adjacent_matrices):
    """TF perturbation pattern → developmental transition 매핑.
    OT weight는 adjacent matrix 전체 평균 → max로 나눠 [0, 1] 정규화.
    """
    print("\n" + "=" * 60)
    print("Step 3: OT-weighted transition attribution (adjacent matrix average)")
    print("=" * 60)

    # Brain organoid transition categories
    transitions_def = {
        'Excitatory_neurogenesis': {
            'description': 'RG->IPC->EN (cortical neuron production)',
            'affected_by': ['ctx_DOWN'],
            'relevant_lineages': ['ctx'],
            'cell_type_pairs': [('RG', 'IPC'), ('IPC', 'EN'), ('RG', 'EN')],
        },
        'Interneuron_specification': {
            'description': 'RG->IN (ventral/interneuron path)',
            'affected_by': ['ge_DOWN'],
            'relevant_lineages': ['ge'],
            'cell_type_pairs': [('RG', 'IN'), ('IPC', 'IN')],
        },
        'Progenitor_maintenance': {
            'description': 'RG->RG, IPC self-renewal',
            'affected_by': ['nt_DOWN'],
            'relevant_lineages': ['nt'],
            'cell_type_pairs': [('RG', 'RG'), ('IPC', 'IPC')],
        },
        'Cortical_overproliferation': {
            'description': 'Aberrant cortical expansion',
            'affected_by': ['ctx_UP'],
            'relevant_lineages': ['ctx'],
            'cell_type_pairs': [('RG', 'IPC'), ('IPC', 'EN')],
        },
        'Ventral_overproliferation': {
            'description': 'Aberrant ventral expansion',
            'affected_by': ['ge_UP'],
            'relevant_lineages': ['ge'],
            'cell_type_pairs': [('RG', 'IN')],
        },
    }

    # Compute raw OT weights from adjacent matrices
    print(f"\n  Using {len(adjacent_matrices)} adjacent transition matrices for OT weights")
    print("\n  Raw OT transition weights (mean across adjacent matrices):")
    for trans_name, trans_info in transitions_def.items():
        ot_weight_raw, pair_details = compute_ot_weights_from_adjacent(
            adjacent_matrices, trans_info['cell_type_pairs'])
        trans_info['ot_weight_raw'] = ot_weight_raw
        print(f"    {trans_name}: OT_weight_raw={ot_weight_raw:.4f} ({', '.join(pair_details)})")

    # Normalize OT weights: divide by max so values fall in [0, 1]
    max_raw_weight = max(t['ot_weight_raw'] for t in transitions_def.values())
    if max_raw_weight > 0:
        for trans_info in transitions_def.values():
            trans_info['ot_weight'] = trans_info['ot_weight_raw'] / max_raw_weight
    else:
        for trans_info in transitions_def.values():
            trans_info['ot_weight'] = 1.0

    print(f"\n  OT weight normalization: dividing by max raw weight = {max_raw_weight:.4f}")
    print("  Normalized OT weights:")
    for trans_name, trans_info in transitions_def.items():
        print(f"    {trans_name}: raw={trans_info['ot_weight_raw']:.4f} -> normalized={trans_info['ot_weight']:.4f}")

    # Assign TFs to transitions
    tf_to_transitions = {}
    transition_tfs = {t: [] for t in transitions_def}

    for gene, pats in patterns.items():
        tf_to_transitions[gene] = []
        for trans_name, trans_info in transitions_def.items():
            if any(p in trans_info['affected_by'] for p in pats):
                tf_to_transitions[gene].append(trans_name)
                transition_tfs[trans_name].append(gene)

    print("\n  TF attribution:")
    for trans_name, tfs in transition_tfs.items():
        desc = transitions_def[trans_name]['description']
        ot_w = transitions_def[trans_name]['ot_weight']
        ot_r = transitions_def[trans_name]['ot_weight_raw']
        print(f"    {trans_name} ({desc}): {len(tfs)} TFs, OT_weight={ot_w:.3f} (raw={ot_r:.4f})")
        if tfs:
            print(f"      TFs: {', '.join(tfs)}")

    return transitions_def, transition_tfs


# ---------------------------------------------------------------------------
# Step 4: Compute vulnerability scores
# ---------------------------------------------------------------------------
def _compute_raw_score_for_tfs(tfs, relevant_lineages, gene_lineage_lor):
    """Helper: compute the raw perturbation score (before OT weighting) for a list of TFs."""
    raw_score = 0.0
    for tf in tfs:
        lor_values = [gene_lineage_lor.get((tf, lin), 0) for lin in relevant_lineages]
        lor = max(lor_values) if lor_values else 0
        if np.isnan(lor):
            continue
        if lor > 1.0:
            weight = RELEVANCE_WEIGHTS['strong_effect']
        elif lor > 0.5:
            weight = RELEVANCE_WEIGHTS['moderate_effect']
        else:
            weight = RELEVANCE_WEIGHTS['weak_effect']
        raw_score += lor * weight
    return raw_score


def compute_vulnerability_scores(transitions_def, transition_tfs, gene_summary):
    """VS = OT_weight_norm x Sum(|lineage_specific_LOR| x relevance_weight).
    각 transition에 relevant한 lineage의 LOR만 사용 — 예: Excitatory_neurogenesis는
    ctx LOR만. 전체 lineage 평균을 쓰면 signal이 희석됨.
    """
    print("\n" + "=" * 60)
    print("Step 4: Computing OT-weighted vulnerability scores (lineage-specific LOR)")
    print("=" * 60)

    # Build gene+lineage -> |log_odds_ratio| lookup
    gene_lineage_lor = {}
    for _, row in gene_summary.iterrows():
        gene_lineage_lor[(row['gene'], row['y'])] = abs(row['log_odds_ratio'])

    vs_scores = {}
    vs_details = {}
    naive_scores = {}

    for trans_name, tfs in transition_tfs.items():
        relevant_lineages = transitions_def[trans_name].get('relevant_lineages', [])
        raw_score = 0.0
        details = []
        for tf in tfs:
            # Use max |LOR| across relevant lineages for this transition
            lor_values = [gene_lineage_lor.get((tf, lin), 0) for lin in relevant_lineages]
            lor = max(lor_values) if lor_values else 0
            if np.isnan(lor):
                continue

            # Relevance weight based on effect size
            if lor > 1.0:
                weight = RELEVANCE_WEIGHTS['strong_effect']
            elif lor > 0.5:
                weight = RELEVANCE_WEIGHTS['moderate_effect']
            else:
                weight = RELEVANCE_WEIGHTS['weak_effect']

            contribution = lor * weight
            raw_score += contribution
            details.append({
                'gene': tf, 'log_odds_ratio': lor,
                'lineages': relevant_lineages,
                'weight': weight, 'contribution': contribution,
            })

        ot_weight = transitions_def[trans_name].get('ot_weight', 1.0)
        final_score = raw_score * ot_weight

        vs_scores[trans_name] = final_score
        naive_scores[trans_name] = raw_score
        vs_details[trans_name] = sorted(details, key=lambda x: -x['contribution'])

        print(f"  {trans_name}: VS={final_score:.2f} "
              f"(raw={raw_score:.2f} x OT_norm={ot_weight:.3f}, {len(details)} TFs, "
              f"lineages={relevant_lineages})")
        for d in vs_details[trans_name][:3]:
            print(f"    {d['gene']} (|LOR|={d['log_odds_ratio']:.2f}, contrib={d['contribution']:.2f})")

    return vs_scores, vs_details, naive_scores


# ---------------------------------------------------------------------------
# Step 5: Permutation test for VS significance
# ---------------------------------------------------------------------------
def permutation_test(transitions_def, transition_tfs, gene_summary,
                     n_permutations=N_PERMUTATIONS):
    """Permutation test: TF-to-transition assignment를 shuffle.
    Transition별 TF 수는 유지, OT weight는 고정 — gene identity만 검증.
    Kidney pipeline과 동일한 로직.
    """
    print("\n" + "=" * 60)
    print(f"Step 5: Permutation test (n={n_permutations})")
    print("=" * 60)

    # Build gene+lineage -> |log_odds_ratio| lookup
    gene_lineage_lor = {}
    for _, row in gene_summary.iterrows():
        gene_lineage_lor[(row['gene'], row['y'])] = abs(row['log_odds_ratio'])

    # Collect all unique TFs assigned to any transition
    all_assigned_tfs = []
    for tfs in transition_tfs.values():
        all_assigned_tfs.extend(tfs)
    # A TF can appear in multiple transitions; we need the full pool with repeats
    # to preserve the total assignment count
    all_assigned_tfs_array = np.array(all_assigned_tfs)

    # Record sizes and order
    trans_names = list(transition_tfs.keys())
    trans_sizes = [len(transition_tfs[t]) for t in trans_names]

    # Observed VS (without OT weight, then multiply by OT weight)
    observed_vs = {}
    for trans_name in trans_names:
        relevant_lineages = transitions_def[trans_name].get('relevant_lineages', [])
        raw = _compute_raw_score_for_tfs(
            transition_tfs[trans_name], relevant_lineages, gene_lineage_lor)
        ot_weight = transitions_def[trans_name].get('ot_weight', 1.0)
        observed_vs[trans_name] = raw * ot_weight

    # Permutation loop
    perm_counts = {t: 0 for t in trans_names}
    rng = np.random.default_rng(42)

    print(f"  Running {n_permutations} permutations...", end='', flush=True)
    for i in range(n_permutations):
        if (i + 1) % 2500 == 0:
            print(f" {i+1}", end='', flush=True)

        # Shuffle the full pool of TF assignments
        shuffled = rng.permutation(all_assigned_tfs_array)

        # Split back into per-transition groups (same sizes as original)
        idx = 0
        for j, trans_name in enumerate(trans_names):
            size = trans_sizes[j]
            perm_tfs = list(shuffled[idx:idx + size])
            idx += size

            relevant_lineages = transitions_def[trans_name].get('relevant_lineages', [])
            raw = _compute_raw_score_for_tfs(perm_tfs, relevant_lineages, gene_lineage_lor)
            ot_weight = transitions_def[trans_name].get('ot_weight', 1.0)
            perm_vs = raw * ot_weight

            if perm_vs >= observed_vs[trans_name]:
                perm_counts[trans_name] += 1

    print(" done.")

    # Compute empirical p-values
    p_values = {}
    for trans_name in trans_names:
        p_values[trans_name] = perm_counts[trans_name] / n_permutations

    print("\n  Permutation test results:")
    for trans_name in trans_names:
        obs = observed_vs[trans_name]
        p = p_values[trans_name]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {trans_name}: VS={obs:.2f}, p={p:.4f} ({sig})")

    return p_values


# ---------------------------------------------------------------------------
# Fig 5: CRISPRi perturbation heatmap
# ---------------------------------------------------------------------------
def plot_fig5(gene_summary):
    """Fig 5: TF perturbation effect heatmap (log odds ratio)."""
    print("\n" + "=" * 60)
    print("Generating Fig 5: CRISPRi perturbation heatmap")
    print("=" * 60)

    # Pivot: gene x lineage -> log_odds_ratio
    pivot = gene_summary.pivot(index='gene', columns='y', values='log_odds_ratio')
    pivot = pivot.reindex(columns=['ctx', 'ge', 'nt'])

    # Sort by max absolute effect
    pivot['max_abs'] = pivot.abs().max(axis=1)
    pivot = pivot.sort_values('max_abs', ascending=False).drop('max_abs', axis=1)

    # Rename columns for display
    pivot.columns = ['Cortical\n(dorsal)', 'Ganglionic\neminence\n(ventral)', 'Neural\ntube']

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=-3, vmax=3,
                ax=ax, annot=True, fmt='.1f', linewidths=0.5,
                cbar_kws={'label': 'Log odds ratio (CMH test)', 'shrink': 0.6})
    ax.set_title('CRISPRi TF knockdown effects on brain organoid lineages\n(Fleck et al. 2022)',
                 fontsize=12, pad=15)
    ax.set_ylabel('Transcription factor')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig5_perturbation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig5_perturbation_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 6: Vulnerable transitions (with p-values)
# ---------------------------------------------------------------------------
def plot_fig6(vs_scores, vs_details, transitions_def, p_values):
    """Fig 6: Vulnerability score per developmental transition, annotated with p-values."""
    print("\n" + "=" * 60)
    print("Generating Fig 6: Vulnerable transitions (brain)")
    print("=" * 60)

    sorted_trans = sorted(vs_scores.items(), key=lambda x: x[1], reverse=True)
    trans_names = [t[0] for t in sorted_trans]
    scores = [t[1] for t in sorted_trans]

    colors = {
        'Excitatory_neurogenesis': '#e74c3c',
        'Interneuron_specification': '#3498db',
        'Progenitor_maintenance': '#9b59b6',
        'Cortical_overproliferation': '#f39c12',
        'Ventral_overproliferation': '#2ecc71',
    }
    bar_colors = [colors.get(t, '#95a5a6') for t in trans_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

    ax = axes[0]
    y_pos = range(len(trans_names))
    ax.barh(y_pos, scores, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t.replace('_', '\n') for t in trans_names], fontsize=10)
    ax.set_xlabel('Vulnerability Score = OT weight (norm) x Sum(|LOR| x relevance weight)', fontsize=10)
    ax.set_title('Developmental transitions vulnerable to\nTF perturbation in brain organoids',
                 fontsize=12)
    ax.invert_yaxis()

    # Add annotations with OT weight and p-value
    for i, (trans, score) in enumerate(sorted_trans):
        n_tfs = len(vs_details[trans])
        ot_w = transitions_def[trans].get('ot_weight', 1.0)
        p = p_values.get(trans, np.nan)
        if p < 0.001:
            p_str = "p<0.001"
        elif p < 0.01:
            p_str = f"p={p:.3f}"
        elif p < 0.05:
            p_str = f"p={p:.3f}"
        else:
            p_str = f"p={p:.3f}"
        ax.text(score + max(scores) * 0.02 if max(scores) > 0 else 0.1, i,
                f'{n_tfs} TFs, OT={ot_w:.3f}, {p_str}',
                va='center', fontsize=8, color='#555')

    # Panel B: TF table
    ax2 = axes[1]
    ax2.axis('off')

    table_data = []
    table_colors = []
    for trans, _ in sorted_trans[:4]:
        for d in vs_details[trans][:3]:
            table_data.append([
                trans.replace('_', '\n'),
                d['gene'],
                f"{d['log_odds_ratio']:.2f}",
                f"{d['contribution']:.2f}",
            ])
            table_colors.append(colors.get(trans, '#95a5a6'))

    if table_data:
        table = ax2.table(
            cellText=table_data,
            colLabels=['Transition', 'TF', '|LOR|', 'Contrib.'],
            loc='center', cellLoc='center',
            colWidths=[0.42, 0.18, 0.18, 0.18])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Style header row
        for col_idx in range(4):
            cell = table[0, col_idx]
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold', fontsize=10)
            cell.set_edgecolor('white')

        # Style data rows with alternating tint and colored transition column
        for i, color in enumerate(table_colors):
            row = i + 1
            # Transition column: use the transition color with transparency
            table[row, 0].set_facecolor(color + '40')
            table[row, 0].set_text_props(fontsize=9, fontweight='bold')
            # Other columns: light alternating background
            bg = '#f9f9f9' if i % 2 == 0 else '#ffffff'
            for col_idx in range(1, 4):
                table[row, col_idx].set_facecolor(bg)
                table[row, col_idx].set_text_props(fontsize=10)
            # Set edge color for all columns
            for col_idx in range(4):
                table[row, col_idx].set_edgecolor('#dddddd')

        ax2.set_title('Top contributing TFs', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig6_vulnerable_transitions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig6_vulnerable_transitions.png")


# ---------------------------------------------------------------------------
# Fig 7: Naive vs OT-weighted comparison
# ---------------------------------------------------------------------------
def plot_fig7(vs_scores, naive_scores, transitions_def):
    """Naive vs OT-weighted VS 비교.
    Left: grouped bar chart, Right: rank change table.
    OT weighting이 실제로 ranking을 바꾸는지가 핵심 — 안 바꾸면 OT의 부가가치 없음.
    """
    print("\n" + "=" * 60)
    print("Generating Fig 7: Naive vs OT-weighted comparison")
    print("=" * 60)

    # Sort by OT-weighted score
    sorted_by_ot = sorted(vs_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_by_naive = sorted(naive_scores.items(), key=lambda x: x[1], reverse=True)

    # Build rank lookups
    naive_rank = {name: i + 1 for i, (name, _) in enumerate(sorted_by_naive)}
    ot_rank = {name: i + 1 for i, (name, _) in enumerate(sorted_by_ot)}

    # Order transitions by OT-weighted rank for the chart
    trans_names = [t[0] for t in sorted_by_ot]
    ot_values = [vs_scores[t] for t in trans_names]
    naive_values = [naive_scores[t] for t in trans_names]

    colors_map = {
        'Excitatory_neurogenesis': '#e74c3c',
        'Interneuron_specification': '#3498db',
        'Progenitor_maintenance': '#9b59b6',
        'Cortical_overproliferation': '#f39c12',
        'Ventral_overproliferation': '#2ecc71',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.5, 1]})

    # --- Left panel: grouped horizontal bar chart ---
    ax = axes[0]
    n = len(trans_names)
    bar_height = 0.35
    y_pos = np.arange(n)

    # Naive bars (gray)
    ax.barh(y_pos - bar_height / 2, naive_values, bar_height,
            color='#bdc3c7', edgecolor='white', label='Naive (no OT weight)')
    # OT-weighted bars (colored)
    ot_bar_colors = [colors_map.get(t, '#95a5a6') for t in trans_names]
    ax.barh(y_pos + bar_height / 2, ot_values, bar_height,
            color=ot_bar_colors, edgecolor='white', label='OT-weighted')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([t.replace('_', '\n') for t in trans_names], fontsize=10)
    ax.set_xlabel('Vulnerability Score', fontsize=11)
    ax.set_title('Effect of OT weighting on vulnerability ranking', fontsize=12)
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=10)

    # Annotate OT weight on each bar
    for i, t in enumerate(trans_names):
        ot_w = transitions_def[t].get('ot_weight', 1.0)
        ax.text(max(max(ot_values), max(naive_values)) * 0.02, i + bar_height / 2,
                f' OT={ot_w:.3f}', va='center', fontsize=8, color='#333', style='italic')

    # --- Right panel: rank change table ---
    ax2 = axes[1]
    ax2.axis('off')

    table_data = []
    row_colors = []
    for t in trans_names:
        nr = naive_rank[t]
        otr = ot_rank[t]
        change = nr - otr  # positive = moved up with OT weighting
        if change > 0:
            arrow = f"+{change} (up)"
        elif change < 0:
            arrow = f"{change} (down)"
        else:
            arrow = "-- (same)"
        table_data.append([
            t.replace('_', '\n'),
            str(nr),
            str(otr),
            arrow,
        ])
        row_colors.append(colors_map.get(t, '#95a5a6'))

    table = ax2.table(
        cellText=table_data,
        colLabels=['Transition', 'Naive\nrank', 'OT\nrank', 'Change'],
        loc='center', cellLoc='center',
        colWidths=[0.40, 0.15, 0.15, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header
    for col_idx in range(4):
        cell = table[0, col_idx]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_edgecolor('white')

    # Style data rows
    for i, color in enumerate(row_colors):
        row = i + 1
        table[row, 0].set_facecolor(color + '30')
        table[row, 0].set_text_props(fontsize=9, fontweight='bold')
        bg = '#f9f9f9' if i % 2 == 0 else '#ffffff'
        for col_idx in range(1, 4):
            table[row, col_idx].set_facecolor(bg)
            table[row, col_idx].set_text_props(fontsize=10)
        for col_idx in range(4):
            table[row, col_idx].set_edgecolor('#dddddd')

    ax2.set_title('Rank changes: Naive -> OT-weighted', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig7_naive_vs_ot_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig7_naive_vs_ot_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Brain Organoid: CRISPRi + Transport Map Integration")
    print("=" * 60)

    # Step 1: Load
    raw_df, gene_summary = load_perturbation_data()

    # Step 2: Classify
    patterns = classify_patterns(gene_summary)

    # Step 3: Load adjacent transition matrices and attribute
    adjacent_matrices, full_matrix = load_adjacent_transition_matrices()

    if not adjacent_matrices:
        print("  ERROR: No transition matrices found. Run 02_moscot_transport_brain.py first.")
        return

    print(f"\n  Loaded {len(adjacent_matrices)} adjacent matrices + 1 full trajectory matrix")

    transitions_def, transition_tfs = attribute_to_transitions(patterns, adjacent_matrices)

    # Step 4: Vulnerability scores (returns both OT-weighted and naive)
    vs_scores, vs_details, naive_scores = compute_vulnerability_scores(
        transitions_def, transition_tfs, gene_summary)

    # Step 5: Permutation test
    p_values = permutation_test(transitions_def, transition_tfs, gene_summary)

    # Print summary with p-values
    print("\n" + "=" * 60)
    print("Summary: Vulnerability scores with statistical significance")
    print("=" * 60)
    for trans, score in sorted(vs_scores.items(), key=lambda x: -x[1]):
        p = p_values.get(trans, np.nan)
        naive = naive_scores[trans]
        ot_w = transitions_def[trans].get('ot_weight', 1.0)
        ot_r = transitions_def[trans].get('ot_weight_raw', 0.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {trans}: VS={score:.2f} (naive={naive:.2f}), "
              f"OT={ot_w:.3f} (raw={ot_r:.4f}), p={p:.4f} {sig}")

    # Figures
    plot_fig5(gene_summary)
    plot_fig6(vs_scores, vs_details, transitions_def, p_values)
    plot_fig7(vs_scores, naive_scores, transitions_def)

    # Save with p-values and OT normalization info
    vs_df = pd.DataFrame([
        {'transition': t, 'vulnerability_score': s,
         'naive_score': naive_scores[t],
         'n_tfs': len(vs_details[t]),
         'ot_weight_raw': transitions_def[t].get('ot_weight_raw', 0.0),
         'ot_weight_normalized': transitions_def[t].get('ot_weight', 1.0),
         'p_value': p_values.get(t, np.nan),
         'significant': p_values.get(t, 1.0) < 0.05,
         'top_tfs': ', '.join([d['gene'] for d in vs_details[t][:5]])}
        for t, s in sorted(vs_scores.items(), key=lambda x: -x[1])
    ])
    vs_df.to_csv(os.path.join(PROCESSED_DIR, 'vulnerability_scores.csv'), index=False)
    print(f"\n  Saved: vulnerability_scores.csv")

    print("\n" + "=" * 60)
    print("Brain organoid perturbation integration complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
