"""
03_rsa_integration.py — Perturbation과 developmental trajectory 통합 (Aim 2, Kidney)

핵심 질문: CRISPR screening에서 특정 유전자가 depleted될 때,
그 효과가 어떤 developmental transition에서 가장 크게 나타나는가?

Ungricht et al.의 RSA score (perturbation effect size)를
02에서 만든 OT transition map 위에 매핑하여 vulnerability score를 계산.
VS = OT_weight_norm x SUM(|RSA| x disease_weight)

Input:  data/raw/mmc2.xlsx (Supplementary Table S1)
        data/processed/transition_matrix_21_28.csv, 28_35.csv (adjacent)
Output: figures/fig5-7, vulnerability_scores.csv

통계적 보강:
- OT weight: adjacent matrix 평균 사용 (full trajectory 단일 matrix보다 stage-specific)
- OT weight normalization: [0, 1] 범위로 정규화
- Permutation test (n=10,000): VS의 통계적 유의성 검증
- Fig 7: naive vs OT-weighted 비교 — OT가 ranking을 바꾸는지 확인
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
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
FIGURE_DIR = os.path.join(PROJECT_DIR, 'figures')

EXCEL_PATH = os.path.join(RAW_DIR, 'mmc2.xlsx')

# Disease category weights — CAKUT/Ciliopathy는 congenital kidney defect의
# 핵심 범주이므로 full weight, CKD는 developmental origin이 간접적, Other는 baseline
DISEASE_WEIGHTS = {
    'CAKUT': 1.0,
    'Ciliopathy': 1.0,
    'CKD': 0.7,
    'Other': 0.5,
}

# Adjacent transition matrix files (day21->28, day28->35)
ADJACENT_MATRIX_FILES = [
    'transition_matrix_21_28.csv',
    'transition_matrix_28_35.csv',
]

N_PERMUTATIONS = 10000


# ---------------------------------------------------------------------------
# Step 1: Load and parse RSA scores from Supplementary Table S1
# ---------------------------------------------------------------------------
def load_rsa_scores():
    """Load RSA scores from S1B and hit annotations from S1D."""
    print("=" * 60)
    print("Step 1: Loading RSA scores from Table S1")
    print("=" * 60)

    # --- S1B: Full screen results (RSA scores) ---
    s1b = pd.read_excel(EXCEL_PATH, sheet_name='S1B_Screen results', header=2)
    s1b = s1b.rename(columns={s1b.columns[0]: 'gene_id', s1b.columns[1]: 'symbol'})
    s1b = s1b.dropna(subset=['symbol'])
    s1b['symbol'] = s1b['symbol'].astype(str)

    # Extract key RSA score columns (Q1 = RSA score, lower = more depleted)
    # KOd0 axis
    rsa_cols_kod0 = [c for c in s1b.columns if 'KOd0' in str(c) and '.Q1' in str(c)]
    # KOd14 axis
    rsa_cols_kod14 = [c for c in s1b.columns if 'KOd14' in str(c) and '.Q1' in str(c)]

    print(f"  S1B: {len(s1b)} genes")
    print(f"  KOd0 RSA columns ({len(rsa_cols_kod0)}): {rsa_cols_kod0[:5]}...")
    print(f"  KOd14 RSA columns ({len(rsa_cols_kod14)}): {rsa_cols_kod14[:5]}...")

    # --- S1D: Hit list with cell type enrichment/depletion categories ---
    s1d = pd.read_excel(EXCEL_PATH, sheet_name='S1D_Hit list 2', header=3)
    s1d = s1d.rename(columns={s1d.columns[0]: 'gene_id', s1d.columns[1]: 'symbol'})
    s1d = s1d.dropna(subset=['symbol'])
    s1d['symbol'] = s1d['symbol'].astype(str)

    print(f"  S1D: {len(s1d)} hit genes")
    print(f"  S1D columns: {list(s1d.columns)}")

    return s1b, s1d, rsa_cols_kod0, rsa_cols_kod14


# ---------------------------------------------------------------------------
# Step 2: Extract disease gene lists
# ---------------------------------------------------------------------------
def extract_disease_genes(s1d):
    """Extract CAKUT, ciliopathy, and other disease gene categories from S1D."""
    print("\n" + "=" * 60)
    print("Step 2: Extracting disease gene categories")
    print("=" * 60)

    disease_genes = {}

    # CAKUT genes (columns from S1D)
    cakut_col = [c for c in s1d.columns if 'CAKUT' in str(c) or 'human mutations' in str(c).lower()]
    if cakut_col:
        cakut = s1d[s1d[cakut_col[0]].notna()]['symbol'].tolist()
        disease_genes['CAKUT'] = cakut
        print(f"  CAKUT genes: {len(cakut)}")

    # Ciliopathy genes
    cilio_col = [c for c in s1d.columns if 'CILIOPATHY' in str(c).upper() or 'ciliar' in str(c).lower()]
    if cilio_col:
        cilio = s1d[s1d[cilio_col[0]].notna()]['symbol'].tolist()
        disease_genes['Ciliopathy'] = cilio
        print(f"  Ciliopathy genes: {len(cilio)}")

    # Mouse model kidney defect
    mouse_col = [c for c in s1d.columns if 'mouse' in str(c).lower()]
    if mouse_col:
        mouse = s1d[s1d[mouse_col[0]].notna()]['symbol'].tolist()
        disease_genes['Mouse_kidney'] = mouse
        print(f"  Mouse kidney defect genes: {len(mouse)}")

    # Manually defined key disease genes (from proposal)
    disease_genes['Key_CAKUT'] = [
        'BMP4', 'CHD1L', 'DSTYK', 'EYA1', 'FGFR2', 'FOXC2', 'HNF1B',
        'JAG1', 'KAT6A', 'LRP2', 'NPHP3', 'PAX2', 'ROBO2', 'SIX1',
        'SIX2', 'SALL4', 'TFAP2A', 'WNT4', 'CCDC170', 'MYH7B'
    ]
    disease_genes['Key_Notch'] = ['JAG1', 'NOTCH2', 'RBPJ', 'PSENEN', 'ADAM10', 'NCSTN']
    disease_genes['Key_Ciliopathy'] = ['KIF3A', 'OFD1', 'PIBF1', 'CEP83', 'INTU', 'C2CD3', 'SCLT1']

    return disease_genes


# ---------------------------------------------------------------------------
# Step 3: Classify hit genes by enrichment/depletion pattern
# ---------------------------------------------------------------------------
def classify_hit_patterns(s1d):
    """Classify each hit gene by its depletion/enrichment pattern in TECs vs Stroma."""
    print("\n" + "=" * 60)
    print("Step 3: Classifying hit gene patterns")
    print("=" * 60)

    patterns = {}
    pattern_cols = {
        'Stroma_DOWN': 'Stroma DOWN',
        'Both_DOWN': 'both DOWN',
        'TEC_DOWN': 'TECs DOWN',
        'Stroma_UP': 'Stroma UP',
        'Both_UP': 'both UP',
        'TEC_UP': 'TECs UP',
    }

    for gene_symbol in s1d['symbol']:
        row = s1d[s1d['symbol'] == gene_symbol].iloc[0]
        gene_pattern = []
        for key, col in pattern_cols.items():
            if col in s1d.columns and pd.notna(row.get(col)):
                gene_pattern.append(key)
        if gene_pattern:
            patterns[gene_symbol] = gene_pattern

    # Summarize
    pattern_counts = {}
    for gene, pats in patterns.items():
        for p in pats:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

    print("  Pattern distribution:")
    for pat, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"    {pat}: {count} genes")

    return patterns


# ---------------------------------------------------------------------------
# Helper: Load adjacent transition matrices
# ---------------------------------------------------------------------------
def load_adjacent_transition_matrices():
    """
    Load adjacent-interval transition matrices (day21->28, day28->35).
    Returns a list of DataFrames, one per adjacent interval.
    """
    matrices = []
    for fname in ADJACENT_MATRIX_FILES:
        path = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(path):
            mat = pd.read_csv(path, index_col=0)
            matrices.append(mat)
            print(f"  Loaded adjacent matrix: {fname} {mat.shape}")
        else:
            print(f"  WARNING: Adjacent matrix not found: {fname}")
    return matrices


# ---------------------------------------------------------------------------
# Step 4: OT-weighted transition attribution (Approach C, improved)
# ---------------------------------------------------------------------------
def attribute_to_transitions(patterns, adjacent_matrices):
    """
    RSA depletion/enrichment pattern을 developmental transition에 매핑.
    OT weight는 adjacent matrix 평균 — full trajectory matrix 하나보다
    stage-specific dynamics를 더 잘 반영. max로 나눠서 [0, 1] 정규화.
    """
    print("\n" + "=" * 60)
    print("Step 4: OT-weighted transition attribution (adjacent matrices)")
    print("=" * 60)

    # Transition definitions: cell_type pairs are used to look up OT probabilities
    transitions = {
        'Epithelial_differentiation': {
            'description': 'PTEC->eTEC, eTEC maintenance',
            'affected_by': ['TEC_DOWN', 'Both_DOWN'],
            'cell_type_pairs': [('PTEC', 'eTEC'), ('eTEC', 'eTEC'), ('DTEC', 'DTEC')],
        },
        'Stromal_maturation': {
            'description': 'S2->aS, stromal expansion',
            'affected_by': ['Stroma_DOWN', 'Both_DOWN'],
            'cell_type_pairs': [('S2', 'aS'), ('S1', 'aS'), ('aS', 'aS')],
        },
        'Podocyte_specification': {
            'description': 'eP->P, podocyte commitment',
            'affected_by': ['Both_DOWN'],
            'cell_type_pairs': [('eP', 'P'), ('eP', 'eP'), ('P', 'P')],
        },
        'TEC_overproliferation': {
            'description': 'Aberrant TEC expansion',
            'affected_by': ['TEC_UP', 'Both_UP'],
            'cell_type_pairs': [('PTEC', 'eTEC'), ('eTEC', 'eTEC')],
        },
        'Stromal_overproliferation': {
            'description': 'Aberrant stromal expansion',
            'affected_by': ['Stroma_UP', 'Both_UP'],
            'cell_type_pairs': [('S2', 'aS'), ('aS', 'aS'), ('S1', 'S1')],
        },
    }

    # Compute OT-derived weight for each transition category:
    # For each cell-type pair, compute the mean probability across adjacent matrices,
    # then sum across all pairs for that transition.
    print(f"\n  Computing OT weights from {len(adjacent_matrices)} adjacent matrices...")
    print("  OT transition weights (mean across adjacent intervals):")

    for trans_name, trans_info in transitions.items():
        ot_weight = 0.0
        pair_details = []
        for src, tgt in trans_info['cell_type_pairs']:
            pair_probs = []
            for mat in adjacent_matrices:
                if src in mat.index and tgt in mat.columns:
                    pair_probs.append(mat.loc[src, tgt])
            if pair_probs:
                mean_prob = np.mean(pair_probs)
                ot_weight += mean_prob
                pair_details.append(f"{src}->{tgt}={mean_prob:.3f}")
            else:
                pair_details.append(f"{src}->{tgt}=N/A")
        trans_info['ot_weight_raw'] = ot_weight
        pair_details_str = ', '.join(pair_details)
        print(f"    {trans_name}: raw_OT_weight={ot_weight:.4f} ({pair_details_str})")

    # Normalize OT weights to [0, 1]
    max_ot = max(t['ot_weight_raw'] for t in transitions.values())
    if max_ot > 0:
        for trans_name, trans_info in transitions.items():
            trans_info['ot_weight'] = trans_info['ot_weight_raw'] / max_ot
    else:
        for trans_name, trans_info in transitions.items():
            trans_info['ot_weight'] = 1.0

    print(f"\n  Normalized OT weights (divided by max={max_ot:.4f}):")
    for trans_name, trans_info in transitions.items():
        print(f"    {trans_name}: raw={trans_info['ot_weight_raw']:.4f}  "
              f"normalized={trans_info['ot_weight']:.4f}")

    # Assign genes to transitions
    gene_to_transitions = {}
    transition_genes = {t: [] for t in transitions}

    for gene, pats in patterns.items():
        gene_to_transitions[gene] = []
        for trans_name, trans_info in transitions.items():
            if any(p in trans_info['affected_by'] for p in pats):
                gene_to_transitions[gene].append(trans_name)
                transition_genes[trans_name].append(gene)

    print("\n  Gene attribution summary:")
    for trans_name, genes in transition_genes.items():
        desc = transitions[trans_name]['description']
        ot_w = transitions[trans_name]['ot_weight']
        print(f"  {trans_name} ({desc}): {len(genes)} genes, OT_weight_norm={ot_w:.4f}")

    return transitions, transition_genes, gene_to_transitions


# ---------------------------------------------------------------------------
# Step 5: Compute vulnerability scores
# ---------------------------------------------------------------------------
def compute_vulnerability_scores(transitions, transition_genes, disease_genes, s1b, rsa_cols_kod0):
    """
    VS(transition) = OT_weight_norm x SUM(|RSA| x disease_weight).
    OT weight가 없으면 모든 transition이 동등 — biologically minor한
    transition에 과대 귀속되는 문제를 OT normalization으로 방지.
    """
    print("\n" + "=" * 60)
    print("Step 5: Computing OT-weighted vulnerability scores")
    print("=" * 60)

    # Build gene -> disease category mapping
    gene_disease_cat = {}
    for cat, genes in disease_genes.items():
        if cat.startswith('Key_'):
            continue  # Skip manually curated sublists
        for g in genes:
            if g not in gene_disease_cat:
                gene_disease_cat[g] = cat

    # Get mean RSA score per gene (average across KOd0 conditions)
    s1b_indexed = s1b.set_index('symbol')
    rsa_numeric = s1b_indexed[rsa_cols_kod0].apply(pd.to_numeric, errors='coerce')
    gene_mean_rsa = rsa_numeric.mean(axis=1)

    # Compute vulnerability scores
    vs_scores = {}
    vs_naive = {}   # Naive scores (no OT weight) for comparison
    vs_details = {}

    for trans_name, genes in transition_genes.items():
        raw_score = 0.0
        details = []
        for gene in genes:
            # Get RSA effect size
            rsa = gene_mean_rsa.get(gene, 0)
            if pd.isna(rsa):
                continue

            # Get disease weight
            cat = gene_disease_cat.get(gene, 'Other')
            weight = DISEASE_WEIGHTS.get(cat, DISEASE_WEIGHTS['Other'])

            contribution = abs(rsa) * weight
            raw_score += contribution
            details.append({
                'gene': gene,
                'rsa': rsa,
                'category': cat,
                'weight': weight,
                'contribution': contribution,
            })

        # Naive score (no OT weighting)
        vs_naive[trans_name] = raw_score

        # Apply normalized OT weight
        ot_weight = transitions[trans_name].get('ot_weight', 1.0)
        final_score = raw_score * ot_weight

        vs_scores[trans_name] = final_score
        vs_details[trans_name] = sorted(details, key=lambda x: -x['contribution'])

        n_disease = sum(1 for d in details if d['category'] != 'Other')
        print(f"  {trans_name}: VS={final_score:.1f} "
              f"(naive={raw_score:.1f} x OT_norm={ot_weight:.4f}, "
              f"{len(details)} genes, {n_disease} disease-annotated)")

        # Top contributors
        for d in vs_details[trans_name][:3]:
            print(f"    {d['gene']} (RSA={d['rsa']:.2f}, {d['category']}, contrib={d['contribution']:.2f})")

    return vs_scores, vs_naive, vs_details


# ---------------------------------------------------------------------------
# Step 6: Permutation test for VS significance
# ---------------------------------------------------------------------------
def permutation_test(transitions, transition_genes, disease_genes, s1b, rsa_cols_kod0,
                     observed_vs, n_permutations=N_PERMUTATIONS):
    """
    Permutation test: gene-to-transition assignment를 shuffle하여
    observed VS가 우연히 나올 확률을 추정.
    Transition별 gene 수는 유지 — 구조적 차이가 아닌 gene identity의 효과만 검증.
    """
    print("\n" + "=" * 60)
    print(f"Step 6: Permutation test (n={n_permutations})")
    print("=" * 60)

    # Build gene -> disease category mapping
    gene_disease_cat = {}
    for cat, genes in disease_genes.items():
        if cat.startswith('Key_'):
            continue
        for g in genes:
            if g not in gene_disease_cat:
                gene_disease_cat[g] = cat

    # Get mean RSA score per gene
    s1b_indexed = s1b.set_index('symbol')
    rsa_numeric = s1b_indexed[rsa_cols_kod0].apply(pd.to_numeric, errors='coerce')
    gene_mean_rsa = rsa_numeric.mean(axis=1)

    # Collect all unique genes across all transitions
    all_genes = list(set(g for genes in transition_genes.values() for g in genes))
    trans_names = list(transition_genes.keys())
    trans_sizes = [len(transition_genes[t]) for t in trans_names]

    # Precompute gene scores: |RSA| x disease_weight for each gene
    gene_scores = {}
    for gene in all_genes:
        rsa = gene_mean_rsa.get(gene, 0)
        if pd.isna(rsa):
            rsa = 0
        cat = gene_disease_cat.get(gene, 'Other')
        weight = DISEASE_WEIGHTS.get(cat, DISEASE_WEIGHTS['Other'])
        gene_scores[gene] = abs(rsa) * weight

    gene_score_arr = np.array([gene_scores[g] for g in all_genes])
    n_genes_total = len(all_genes)

    # OT weights for each transition
    ot_weights = np.array([transitions[t].get('ot_weight', 1.0) for t in trans_names])

    # Count how many permuted VS >= observed VS
    perm_counts = np.zeros(len(trans_names))
    observed_arr = np.array([observed_vs[t] for t in trans_names])

    rng = np.random.default_rng(42)

    print(f"  Running {n_permutations} permutations across {n_genes_total} genes "
          f"and {len(trans_names)} transitions...")

    for i in range(n_permutations):
        if (i + 1) % 2000 == 0:
            print(f"    Permutation {i + 1}/{n_permutations}...")

        # Shuffle gene scores
        shuffled_scores = rng.permutation(gene_score_arr)

        # Assign shuffled genes to transitions (same sizes as original)
        idx = 0
        for j, size in enumerate(trans_sizes):
            perm_raw = shuffled_scores[idx:idx + size].sum()
            perm_vs = perm_raw * ot_weights[j]
            if perm_vs >= observed_arr[j]:
                perm_counts[j] += 1
            idx += size

    # Compute empirical p-values
    p_values = {}
    for j, t in enumerate(trans_names):
        p = perm_counts[j] / n_permutations
        p_values[t] = p
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f"  {t}: p={p:.4f} {sig} "
              f"(observed VS={observed_arr[j]:.1f}, "
              f"exceeded in {int(perm_counts[j])}/{n_permutations} permutations)")

    return p_values


# ---------------------------------------------------------------------------
# Fig 5: Disease gene RSA score heatmap
# ---------------------------------------------------------------------------
def plot_fig5(s1b, rsa_cols_kod0, rsa_cols_kod14, disease_genes):
    """Fig 5: RSA score heatmap for key disease genes."""
    print("\n" + "=" * 60)
    print("Generating Fig 5: Disease gene RSA heatmap")
    print("=" * 60)

    # Select key disease genes present in S1B
    key_genes = list(dict.fromkeys(
        disease_genes.get('Key_CAKUT', []) +
        disease_genes.get('Key_Ciliopathy', []) +
        disease_genes.get('Key_Notch', [])
    ))

    s1b_indexed = s1b.set_index('symbol')
    available = [g for g in key_genes if g in s1b_indexed.index]
    print(f"  Key genes found in S1B: {len(available)}/{len(key_genes)}")

    if not available:
        print("  No genes found. Skipping Fig 5.")
        return

    # Select RSA columns and clean names
    # KOd0 columns
    kod0_display = {}
    for c in rsa_cols_kod0:
        short = c.replace('KOd0_', '').replace('.Q1', '').replace(' (', '\n(').replace(')', ')')
        # Simplify further
        if 'NPC to iPSC' in c:
            short = 'NPC\n(KOd0)'
        elif 'day21' in c and 'Organoid vs. iPSC' in c:
            short = 'd21\n(KOd0)'
        elif 'day35' in c and 'Organoid vs. iPSC' in c:
            short = 'd35\n(KOd0)'
        elif 'TEC vs. iPSC' in c:
            short = 'TEC\n(KOd0)'
        elif 'Stroma vs. iPSC' in c:
            short = 'Stroma\n(KOd0)'
        elif 'TEC vs. NPC' in c:
            short = 'TEC/NPC\n(KOd0)'
        elif 'Stroma vs. NPC' in c:
            short = 'Stroma/NPC\n(KOd0)'
        else:
            continue
        kod0_display[c] = short

    # Build heatmap matrix
    selected_cols = list(kod0_display.keys())
    rsa_matrix = s1b_indexed.loc[available, selected_cols].apply(pd.to_numeric, errors='coerce')
    rsa_matrix.columns = [kod0_display[c] for c in selected_cols]

    # Add disease category annotation
    gene_cats = []
    for g in available:
        if g in disease_genes.get('Key_CAKUT', []):
            gene_cats.append('CAKUT')
        elif g in disease_genes.get('Key_Ciliopathy', []):
            gene_cats.append('Ciliopathy')
        elif g in disease_genes.get('Key_Notch', []):
            gene_cats.append('Notch')
        else:
            gene_cats.append('Other')

    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(available) * 0.4)))

    # Color by category on y-axis
    cat_colors = {'CAKUT': '#e74c3c', 'Ciliopathy': '#3498db', 'Notch': '#2ecc71', 'Other': '#95a5a6'}
    row_colors = [cat_colors.get(c, '#95a5a6') for c in gene_cats]

    g = sns.heatmap(
        rsa_matrix, cmap='RdBu_r', center=0, vmin=-8, vmax=2,
        ax=ax, xticklabels=True, yticklabels=True,
        cbar_kws={'label': 'RSA score (Q1)', 'shrink': 0.6},
        linewidths=0.5
    )

    # Add category color bar on left
    for i, color in enumerate(row_colors):
        ax.add_patch(plt.Rectangle((-0.8, i), 0.6, 1,
                                   fill=True, color=color, transform=ax.transData,
                                   clip_on=False))

    ax.set_title('RSA scores of congenital kidney disease genes\n(Ungricht et al. 2022 CRISPR screening)',
                 fontsize=12, pad=20)
    ax.set_ylabel('')

    # Legend for disease categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors[c], label=c) for c in ['CAKUT', 'Ciliopathy', 'Notch']]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig5_disease_gene_rsa_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig5_disease_gene_rsa_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 6: Vulnerable transition map
# ---------------------------------------------------------------------------
def plot_fig6(vs_scores, vs_details, transitions, p_values=None):
    """Fig 6: Vulnerability score per developmental transition."""
    print("\n" + "=" * 60)
    print("Generating Fig 6: Vulnerable transitions")
    print("=" * 60)

    # Sort transitions by vulnerability score
    sorted_trans = sorted(vs_scores.items(), key=lambda x: x[1], reverse=True)
    trans_names = [t[0] for t in sorted_trans]
    scores = [t[1] for t in sorted_trans]

    # Color by transition type
    colors = {
        'Epithelial_differentiation': '#e74c3c',
        'Stromal_maturation': '#3498db',
        'Podocyte_specification': '#9b59b6',
        'TEC_overproliferation': '#f39c12',
        'Stromal_overproliferation': '#2ecc71',
    }
    bar_colors = [colors.get(t, '#95a5a6') for t in trans_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

    # Panel A: Vulnerability score bar chart
    ax = axes[0]
    y_pos = range(len(trans_names))
    bars = ax.barh(y_pos, scores, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t.replace('_', '\n') for t in trans_names], fontsize=10)
    ax.set_xlabel('Vulnerability Score = OT weight (norm) x SUM(|RSA| x disease weight)', fontsize=10)
    ax.set_title('Developmental transitions vulnerable to\ncongenital kidney disease gene perturbation',
                 fontsize=12)
    ax.invert_yaxis()

    # Add gene count + OT weight + p-value annotations
    for i, (trans, score) in enumerate(sorted_trans):
        n_genes = len(vs_details[trans])
        n_disease = sum(1 for d in vs_details[trans] if d['category'] != 'Other')
        ot_w = transitions[trans].get('ot_weight', 1.0)
        annotation = (f'{n_genes} genes ({n_disease} disease)\n'
                      f'OT weight={ot_w:.3f}')
        if p_values is not None and trans in p_values:
            p = p_values[trans]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            annotation += f'\np={p:.4f} {sig}'
        ax.text(score + max(scores) * 0.02, i, annotation,
                va='center', fontsize=8, color='#555')

    # Panel B: Top contributing genes table
    ax2 = axes[1]
    ax2.axis('off')

    # Build table data
    table_data = []
    table_colors = []
    for trans, _ in sorted_trans[:4]:  # Top 4 transitions
        for d in vs_details[trans][:3]:  # Top 3 genes each
            table_data.append([
                trans.replace('_', '\n'),
                d['gene'],
                f"{d['rsa']:.1f}",
                d['category'],
            ])
            table_colors.append(colors.get(trans, '#95a5a6'))

    if table_data:
        table = ax2.table(
            cellText=table_data,
            colLabels=['Transition', 'Gene', 'RSA', 'Category'],
            loc='center',
            cellLoc='center',
            colWidths=[0.35, 0.20, 0.15, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.6)

        # Color first column by transition
        for i, color in enumerate(table_colors):
            table[i + 1, 0].set_facecolor(color + '30')

        ax2.set_title('Top contributing genes', fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig6_vulnerable_transitions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig6_vulnerable_transitions.png")


# ---------------------------------------------------------------------------
# Fig 7: Naive vs OT-weighted comparison
# ---------------------------------------------------------------------------
def plot_fig7(vs_scores, vs_naive, transitions):
    """
    Fig 7: Side-by-side comparison of naive VS (no OT weighting) vs
    OT-weighted VS, showing how OT weighting changes transition ranking.
    """
    print("\n" + "=" * 60)
    print("Generating Fig 7: Naive vs OT-weighted comparison")
    print("=" * 60)

    trans_names = list(vs_scores.keys())

    # Compute ranks (1 = highest VS)
    naive_ranked = sorted(trans_names, key=lambda t: vs_naive[t], reverse=True)
    ot_ranked = sorted(trans_names, key=lambda t: vs_scores[t], reverse=True)
    naive_rank = {t: i + 1 for i, t in enumerate(naive_ranked)}
    ot_rank = {t: i + 1 for i, t in enumerate(ot_ranked)}

    # Sort by OT-weighted score for display
    display_order = ot_ranked

    colors = {
        'Epithelial_differentiation': '#e74c3c',
        'Stromal_maturation': '#3498db',
        'Podocyte_specification': '#9b59b6',
        'TEC_overproliferation': '#f39c12',
        'Stromal_overproliferation': '#2ecc71',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={'width_ratios': [3, 1]})

    # --- Left panel: side-by-side horizontal bar chart ---
    ax = axes[0]
    y_pos = np.arange(len(display_order))
    bar_height = 0.35

    naive_vals = [vs_naive[t] for t in display_order]
    ot_vals = [vs_scores[t] for t in display_order]

    bars_naive = ax.barh(y_pos + bar_height / 2, naive_vals, bar_height,
                         color='#bdc3c7', edgecolor='white', label='Naive (no OT weight)')
    bars_ot = ax.barh(y_pos - bar_height / 2, ot_vals, bar_height,
                      color=[colors.get(t, '#95a5a6') for t in display_order],
                      edgecolor='white', label='OT-weighted (normalized)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([t.replace('_', '\n') for t in display_order], fontsize=10)
    ax.set_xlabel('Vulnerability Score', fontsize=11)
    ax.set_title('Naive vs OT-weighted vulnerability scores', fontsize=12)
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=9)

    # --- Right panel: rank change table ---
    ax2 = axes[1]
    ax2.axis('off')

    table_data = []
    for t in display_order:
        rank_change = naive_rank[t] - ot_rank[t]
        if rank_change > 0:
            arrow = f'+{rank_change} (up)'
        elif rank_change < 0:
            arrow = f'{rank_change} (down)'
        else:
            arrow = '0 (same)'
        table_data.append([
            t.replace('_', '\n'),
            str(naive_rank[t]),
            str(ot_rank[t]),
            arrow,
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=['Transition', 'Naive\nRank', 'OT\nRank', 'Change'],
        loc='center',
        cellLoc='center',
        colWidths=[0.38, 0.15, 0.15, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    # Color the transition column
    for i, t in enumerate(display_order):
        table[i + 1, 0].set_facecolor(colors.get(t, '#95a5a6') + '30')

    ax2.set_title('Rank change from OT weighting', fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fig7_naive_vs_ot_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig7_naive_vs_ot_comparison.png")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("Aim 2: RSA Score + Transport Map Integration")
    print("(with adjacent-matrix OT weights, normalization, and permutation test)")
    print("=" * 60)

    # Step 1: Load RSA scores
    s1b, s1d, rsa_cols_kod0, rsa_cols_kod14 = load_rsa_scores()

    # Step 2: Extract disease gene lists
    disease_genes = extract_disease_genes(s1d)

    # Step 3: Classify hit patterns
    patterns = classify_hit_patterns(s1d)

    # Step 4: Load adjacent transition matrices and attribute genes to transitions
    print("\n  Loading adjacent transition matrices for OT weights...")
    adjacent_matrices = load_adjacent_transition_matrices()
    if not adjacent_matrices:
        print("  ERROR: No adjacent matrices found. Falling back to full trajectory matrix.")
        full_mat = pd.read_csv(
            os.path.join(PROCESSED_DIR, 'transition_matrix_21_35.csv'), index_col=0
        )
        adjacent_matrices = [full_mat]

    transitions, transition_genes, gene_to_transitions = attribute_to_transitions(
        patterns, adjacent_matrices
    )

    # Step 5: Compute OT-weighted vulnerability scores (with normalized weights)
    vs_scores, vs_naive, vs_details = compute_vulnerability_scores(
        transitions, transition_genes, disease_genes, s1b, rsa_cols_kod0
    )

    # Step 6: Permutation test for significance
    p_values = permutation_test(
        transitions, transition_genes, disease_genes, s1b, rsa_cols_kod0,
        observed_vs=vs_scores, n_permutations=N_PERMUTATIONS
    )

    # Figures
    plot_fig5(s1b, rsa_cols_kod0, rsa_cols_kod14, disease_genes)
    plot_fig6(vs_scores, vs_details, transitions, p_values=p_values)
    plot_fig7(vs_scores, vs_naive, transitions)

    # Save vulnerability scores with p-values
    vs_df = pd.DataFrame([
        {'transition': t,
         'vulnerability_score': s,
         'naive_score': vs_naive[t],
         'ot_weight_raw': transitions[t].get('ot_weight_raw', np.nan),
         'ot_weight_normalized': transitions[t].get('ot_weight', np.nan),
         'p_value': p_values.get(t, np.nan),
         'n_genes': len(vs_details[t]),
         'n_disease_genes': sum(1 for d in vs_details[t] if d['category'] != 'Other'),
         'top_genes': ', '.join([d['gene'] for d in vs_details[t][:5]])}
        for t, s in sorted(vs_scores.items(), key=lambda x: -x[1])
    ])
    vs_df.to_csv(os.path.join(PROCESSED_DIR, 'vulnerability_scores.csv'), index=False)
    print(f"\n  Saved: vulnerability_scores.csv (with p-values and OT weight details)")

    print("\n" + "=" * 60)
    print("Aim 2 analysis complete.")
    print("Figures generated: Fig 5, Fig 6, Fig 7")
    print("Statistical improvements: adjacent OT weights, normalization, permutation test")
    print("=" * 60)


if __name__ == '__main__':
    main()
