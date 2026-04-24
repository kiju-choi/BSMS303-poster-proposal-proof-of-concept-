# Validation Datasets: Process, Results, Assessment

---

## 1. Kidney Organoid — Ungricht et al. (Cell Stem Cell, 2022)

### 1.1 Data Source and Biological Context

**Paper:** Genome-wide screening in human kidney organoids identifies developmental and disease-related aspects of nephrogenesis.

**Experimental design:** iPSC-derived kidney organoids with doxycycline-inducible Cas9. Genome-wide pooled CRISPR screen (18,360 genes) at two editing timepoints (KO^d0 at iPSC stage, KO^d14 after lineage specification). Control organoid scRNA-seq at three collection timepoints.

**Data used:**
- scRNA-seq: 7 H5 files from GEO GSE181595
  - Day 21: 2 replicates
  - Day 28: 3 replicates
  - Day 35: 2 replicates
- Perturbation data: RSA (Redundant siRNA Activity) scores from Supplementary Table mmc2.xlsx
  - 6 readout conditions: NPC, Stroma, Stroma/NPC (at KO^d0); TEC, TEC/NPC (at KO^d0 and KO^d14)
  - RSA score < 0 = depleted, RSA score > 0 = enriched

### 1.2 Step 1: Preprocessing (`01_preprocessing.py`)

**Parameters (reproducing original paper):**
- Gene filter: expressed in >= 3 cells
- Cell filter: min_genes=200, pct_mt < 20%
- Normalization: `sc.pp.normalize_total(target_sum=1e4)` → `sc.pp.log1p()`
- HVG: `sc.pp.highly_variable_genes(min_mean=0.01, max_mean=3, min_disp=0.5)`
- PCA: 40 components
- Neighbors: k=20, 40 PCs
- Clustering: Leiden (resolution tuned to match original 11 clusters)

**Results:**
- Input: 24,781 cells (7 samples)
- Post-QC: **23,360 cells**
- Cell types identified: 11 (eTEC, PTEC, DTEC, LH, CC, P, eP, aS, S1, S2, + minor populations)
- Cell type proportions shift across timepoints consistent with original paper Fig 1C:
  - Day 21: dominated by early progenitors (eTEC, eP)
  - Day 35: increased mature tubular (PTEC, DTEC) and stromal (S1, S2) fractions

**Output:** `adata_preprocessed.h5ad`

### 1.3 Step 2: OT Trajectory Inference (`02_moscot_transport.py`)

**moscot configuration:**
- `TemporalProblem` with sequential policy (adjacent timepoints only)
- Growth rate estimation: `score_genes_for_marginals()` using proliferation/apoptosis gene sets (Schiebinger et al., 2019)
- Epsilon: auto-tuned with sensitivity analysis across [0.001, 0.005, 0.01, 0.05, 0.1]
- Unbalanced marginals: `tau_a`, `tau_b` < 1 (allowing birth/death)

**Transport maps computed:**
- Day 21 → Day 28 (adjacent)
- Day 28 → Day 35 (adjacent)
- Day 21 → Day 35 (full trajectory, chained)

**Key observations:**

1. **Near-identity transition matrices:** The dominant pattern in all three transition matrices is high diagonal probability — most cells at time t are mapped to the same cell type at time t+1. For example, PTEC→PTEC = 0.99, S1→S1 = 0.98, DTEC→DTEC = 0.99. This "near-identity" pattern limits the information content of the trajectory.

2. **Meaningful off-diagonal signals are sparse:** The strongest non-trivial transition is eTEC → PTEC (0.67 in Day 21→28), reflecting proximal tubular differentiation from early tubular progenitors. Other off-diagonal entries are mostly < 0.05.

3. **Epsilon sensitivity:** Transition matrices are relatively stable across epsilon values, confirming the auto-tuned choice is reasonable. Very small epsilon (0.001) produces slightly more concentrated (sparser) mappings, but the overall pattern is unchanged.

4. **Ancestor probability analysis (pull-back):** Pulling back Day 35 PTEC ancestors shows strongest signal in Day 21 eTEC and PTEC populations, consistent with known nephrogenic differentiation. However, the sparse off-diagonal structure limits the resolution of bifurcation analysis.

**Why near-identity?**
- 3 timepoints at 7-day intervals is too sparse for OT. For reference, Schiebinger et al. (2019) used 39 timepoints at 12-hour intervals. With only 3 widely-spaced snapshots, most cells "stay" in their annotated type — intermediate states simply aren't captured.
- This is a data limitation, not a method limitation. The same pipeline on the brain dataset (11 timepoints) yields non-trivial transitions — see Section 2.

**Output:** `adata_with_transport.h5ad`, `transition_matrix_*.csv` (3 files)

### 1.4 Step 3: Perturbation Integration (`03_rsa_integration.py`)

**Perturbation data processing:**
- RSA scores extracted from mmc2.xlsx for congenital kidney disease genes:
  - **CAKUT genes:** BMP4, CHD1L, DSTYK, EYA1, FGFR2, HNF1B, KAT6A, LRP2, MYH7B, NPR3, PAX2, ROBO2, SIX1, SIX2, SALL4, TRAP4, WNT4, CCDC170
  - **Ciliopathy genes:** KIF3A, OFD1, PKD1, CEP83, INTU, CJCIC3, SCLT1, NOTCH2, RFP5, PSEN1N, ADAM10, NCSTN
  - **Notch pathway genes:** JAG1 (from paper main text)

**Hit pattern classification:**
- Each gene's RSA score pattern (UP/DOWN across NPC, TEC, Stroma conditions) is classified into biological transition categories
- Rule-based mapping: e.g., TEC_DOWN genes → mapped to epithelial differentiation transition; Stroma_DOWN → stromal maturation

**OT weight computation:**
- Adjacent matrix average: mean OT weight across Day 21→28 and Day 28→35 matrices for each transition
- Max-normalization: weights scaled to [0, 1] range

**Vulnerability Score (VS):**
$$VS_i = W_i^{norm} \times \sum_{g \in G_i} |RSA_g| \times D_g$$

Where:
- $W_i^{norm}$ = normalized OT weight for transition $i$
- $|RSA_g|$ = absolute RSA score (effect size) of gene $g$
- $D_g$ = disease weight (1.5 for known disease genes, 1.0 otherwise)

**Results:**

| Transition | VS | OT Weight | # Genes (disease) | p-value |
|-----------|-----|-----------|-------------------|---------|
| **Stromal maturation** | **1018** | **1.000** | 10 (18 disease) | **< 0.0001** |
| Epithelial differentiation | 122.5 | 0.471 | 13 (12 disease) | < 0.0001 |
| Podocyte specification | 65.7 | 0.078 | 4 (7 disease) | < 0.0001 |
| Stromal overproliferation | 67.4 | 1.100 | 15 (17 disease) | < 0.0001 |
| TEC overproliferation | 16.6 | 0.451 | 18 (44 disease) | n.s. |

**Statistical testing:**
- Permutation test (n=10,000): Shuffle gene-to-transition assignments, recompute VS, compare observed vs null distribution
- Stromal maturation: p < 0.0001 (observed VS far exceeds null)
- TEC overproliferation: not significant (p > 0.05)

**Naive vs OT-weighted comparison:**
- Without OT weighting (naive = just sum of |RSA|), Epithelial differentiation ranks #1
- With OT weighting, **Stromal maturation moves to #1** (rank change: +1)
- Epithelial differentiation drops to #2 (rank change: -1)
- This rank reversal demonstrates that **trajectory information adds value beyond simple perturbation effect aggregation**

### 1.5 Kidney Dataset — Self-Assessment

**What works:**
- Pipeline runs, produces statistically significant results, and OT weighting changes the vulnerability ranking vs naive aggregation — so trajectory context does add something
- Stromal maturation as top vulnerable transition is biologically plausible (stroma provides niche signals for nephron development)

**What doesn't:**
- Near-identity matrices mean OT weights are dominated by self-transitions — the "trajectory" is barely a trajectory
- RSA scores are bulk, population-level — no way to know which specific cells are affected
- 432 "Both_DOWN" genes produce the same top hits across transitions, so the perturbation readout itself lacks specificity
- Bottom line: **3 timepoints is not enough for OT trajectory inference.** This dataset is more useful as a cautionary example than as a positive result

---

## 2. Brain Organoid — Fleck et al. (Nature, 2022)

### 2.1 Data Source and Biological Context

**Paper:** Inferring and perturbing cell fate regulomes in human brain organoids.

**Experimental design:** Brain organoid scRNA-seq time series (11 timepoints, Day 4–61) capturing the progression from neuroepithelial progenitors through radial glia to mature neuronal subtypes. Separately, CRISPRi knockdown of 20 transcription factors with lineage enrichment readout, plus CROP-seq data (same TFs, per-cell resolution).

**Data used:**
- scRNA-seq: `RNA_data.h5ad` from Zenodo (49,718 cells pre-QC, 11 timepoints)
- Perturbation data: `ST5_CRISPRi_enrichment.xls`
  - 20 TFs: BACH2, BCL11B, E2F2, FOXN4, GLI3, HES1, HOPX, LHX2, MEIS1, MYT1L, NEUROD1, NEUROD6, PAX6, SOX1, SOX5, SOX9, ST18, TBR1, ZEB2, ZFPM2
  - 3 lineage readouts: cortical (ctx), ganglionic eminence (ge), neural tube (nt)
  - Log-odds ratio (LOR) from Cochran-Mantel-Haenszel test

### 2.2 Step 1: Preprocessing (`brain/01_preprocessing_brain.py`)

**Parameters:**
- Used original paper's cell type annotations (Nowakowski classification)
- Gene filter: expressed in >= 3 cells
- Cell filter: min_genes=200, pct_mt < 20%
- Normalization: `sc.pp.normalize_total(target_sum=1e4)` → `sc.pp.log1p()`
- HVG: `sc.pp.highly_variable_genes(n_top_genes=3000)`
- PCA: 50 components
- Neighbors: k=20, 50 PCs

**Results:**
- Post-QC: **44,483 cells**, 8 major cell types
- Cell types: Astro (astrocyte), Choroid, EN (excitatory neuron), IN (interneuron), IPC (intermediate progenitor cell), Meso (mesenchymal), OPC (oligodendrocyte progenitor), RG (radial glia)
- 11 timepoints provide dense temporal coverage of brain organoid development:
  - Early (Day 4–9): dominated by RG and IPC
  - Mid (Day 10–26): EN and IN emergence
  - Late (Day 31–61): mature neuronal types predominate

**Output:** `adata_preprocessed.h5ad`

### 2.3 Step 2: OT Trajectory Inference (`brain/02_moscot_transport_brain.py`)

**moscot configuration:** Same as kidney (TemporalProblem, sequential policy, auto epsilon, unbalanced marginals).

**Transport maps computed:** 10 adjacent transition matrices + 1 full trajectory (Day 4→61)

**Key observations:**

1. **Non-trivial transition patterns:** Unlike kidney, the brain dataset yields **biologically meaningful off-diagonal transitions**:
   - RG → IPC (0.15–0.30 across early timepoints): radial glia giving rise to intermediate progenitors
   - IPC → EN (0.10–0.25 across mid timepoints): intermediate progenitors differentiating into excitatory neurons
   - RG → IN (variable): direct interneuron generation from radial glia

2. **Temporal dynamics are captured:** The transition probabilities change across developmental stages:
   - Early transitions (Day 4→7): high RG self-renewal, low differentiation
   - Mid transitions (Day 16→21): peak IPC→EN differentiation
   - Late transitions (Day 31→61): mostly maintenance/maturation

3. **One clear artifact:** EN→IN at Day 31→61 shows OT weight of 0.711 — almost certainly spurious. The 30-day gap between these timepoints means OT is mapping by expression similarity, and late EN/IN likely converge in expression space. This is an artifact of uneven temporal spacing, not a real transition.

4. **Ancestor probability analysis:** Pulling back Day 61 EN ancestors shows progressive signal from RG (earliest) through IPC (intermediate) to EN (latest), correctly recapitulating the known RG → IPC → EN differentiation cascade in cortical neurogenesis.

**Output:** `adata_with_transport.h5ad`, `transition_matrix_*.csv` (11 files)

### 2.4 Step 3: Perturbation Integration (`brain/03_perturbation_integration_brain.py`)

**Perturbation data processing:**
- CRISPRi enrichment scores (LOR) from ST5 for 20 TFs × 3 lineages
- Each TF's LOR pattern classified into developmental transition categories:
  - **ctx_DOWN** (cortical neuron depletion) → Excitatory neurogenesis disruption
  - **ge_DOWN** (ganglionic eminence depletion) → Interneuron specification disruption
  - **nt_DOWN/UP** (neural tube change) → Progenitor maintenance disruption
  - Bidirectional patterns mapped to multiple transitions

**Lineage-specific LOR integration:**
- Unlike kidney (which used a single RSA value per gene), brain perturbation data has **lineage-resolved** readouts
- For each transition, only the **relevant lineage's LOR** is used:
  - Excitatory neurogenesis: ctx LOR only
  - Interneuron specification: ge LOR only
  - Progenitor maintenance: nt LOR only

**OT weight computation:**
- Adjacent matrix average: mean OT weight across **10 adjacent matrices**
- Max-normalization to [0, 1]

**Results:**

| Transition | VS | OT Weight (norm) | # TFs | p-value |
|-----------|-----|-------------------|-------|---------|
| Progenitor maintenance | 5.66 | 1.000 | 11 | 0.078 |
| **Excitatory neurogenesis** | **2.30** | **0.486** | **8** | **0.0005*** |
| Ventral overproliferation | 0.01 | 0.003 | 6 | 0.005 |
| Interneuron specification | 0.00 | 0.003 | 5 | 0.459 |
| Cortical overproliferation | 0.00 | 0.521 | 0 | 1.000 |

**Key finding — Excitatory neurogenesis is the most statistically significant vulnerable transition (p = 0.0005):**
- Top TFs: **TBR1** (|LOR|=1.48, master regulator of cortical neuron identity), **BACH2** (|LOR|=1.48), **BCL11B** (|LOR|=1.13, corticospinal neuron specification), **ST18** (|LOR|=0.94)
- All known cortical EN regulators — the framework recovers established biology, which is reassuring but not novel

**Progenitor maintenance (VS rank #1 but p = 0.078):**
- Highest raw VS because progenitor self-renewal dominates the OT weight — but borderline significance suggests this partly reflects baseline transition strength, not specific vulnerability
- Top TFs: LHX2 (|LOR|=1.00), SOX9 (|LOR|=0.92)

**Naive vs OT-weighted — no rank change:**
- Unlike kidney, OT weighting doesn't change the ranking here. Why? The brain perturbation data is already lineage-specific (each transition uses only its relevant lineage's LOR), so OT has less to add
- This is actually informative: **OT weighting adds value when the perturbation readout is non-specific** (bulk RSA), not when it's already lineage-resolved

### 2.5 Brain Dataset — Self-Assessment

**What works:**
- 11 timepoints produce non-trivial transitions that 3 timepoints can't — this alone validates the temporal resolution argument
- Excitatory neurogenesis as the top vulnerable transition is biologically plausible, and the TFs (TBR1, BCL11B) are established cortical regulators
- Permutation p = 0.0005 — the framework can achieve genuine statistical rigor

**What doesn't:**
- **OT weighting adds nothing here** — rankings are identical with or without it, because the perturbation data is already lineage-specific. OT weighting matters most when the perturbation readout is non-specific
- **Progenitor maintenance confound:** self-renewal dominates OT weight, inflating VS for progenitor transitions regardless of actual vulnerability
- **EN→IN artifact:** uneven temporal spacing (30-day gap) produces a spurious mapping — identified and excluded, but a reminder that OT is sensitive to study design
- **No novel biology:** TBR1 and BCL11B as cortical neuron regulators is textbook. The framework confirms what's known rather than discovering something new

---

## 3. Cross-Dataset Comparison

### 3.1 Temporal Resolution Effect

| Parameter | Kidney | Brain | Impact |
|-----------|--------|-------|--------|
| Timepoints | 3 (Day 21, 28, 35) | 11 (Day 4–61) | Brain captures intermediate transitions |
| Interval | 7 days uniform | 1–30 days variable | Kidney too coarse for OT |
| Cells | 23,360 | 44,483 | Both sufficient for OT |
| Transition matrices | 2 adjacent | 10 adjacent | Brain has 5x more transition data |
| **Off-diagonal signal** | **Near-identity (< 0.05)** | **Non-trivial (up to 0.30)** | **Critical difference** |

**Takeaway:** Temporal resolution is the single most important factor. Same pipeline, same parameters, same statistics — 3 vs 11 timepoints produces qualitatively different results. Empirical minimum for meaningful OT: **~8–10 timepoints.**

### 3.2 Perturbation Data Specificity Effect

| Parameter | Kidney | Brain |
|-----------|--------|-------|
| Readout type | RSA score (barcode counting) | CRISPRi LOR (lineage enrichment) |
| Specificity | Bulk, population-level | Lineage-specific (3 lineages) |
| # Genes/TFs | ~30 disease genes | 20 TFs |
| OT weighting effect | **Rank reversal** (adds information) | No rank change (already specific) |

**Takeaway:** OT weighting adds the most value when perturbation readout is non-specific. When it's already lineage-resolved, trajectory context has less to contribute. Practical implication: for genome-wide screens with bulk readouts — the most common organoid CRISPR design — OT reinterpretation is most useful.

### 3.3 What Both Datasets Show Together

1. **Generalizable:** Same 3-step pipeline works across organ systems — different cell types, perturbation designs, readout types
2. **Statistically rigorous:** Permutation testing catches real signal (brain EN, p = 0.0005) and correctly flags noise (kidney TEC overproliferation, n.s.)
3. **Proof-of-concept, not discovery:** Neither dataset produced a novel biological finding. The contribution is showing that population-level perturbation readouts *can* be reinterpreted in a trajectory context — and clarifying *when* that reinterpretation actually adds value

---

