# BSMS303-poster-proposal-proof-of-concept-

# OT-Based Developmental Vulnerability Mapping in Organoids

**When during development do genetic perturbations do the most damage?**

> Course project for Human Embryology, 2026 Spring

---

## Overview

Organoid CRISPR screens tell us *which* genes disrupt development — but not *when* during development those disruptions hit hardest. This project integrates **OT-based trajectory inference** (moscot) with **perturbation effect sizes** to map vulnerability onto specific developmental transitions.

The core idea: if OT gives us the probability that cell state A becomes cell state B, and a CRISPR screen tells us gene X depletes cell state B, we can ask *which transition* is most disrupted — and whether that answer changes when trajectory context is included.

Validated on two independent organ systems:
- **Kidney organoid** — genome-wide CRISPR screen (Ungricht et al., Cell Stem Cell, 2022)
- **Brain organoid** — CRISPRi of 20 TFs (Fleck et al., Nature, 2022)

For detailed results, see [`validation_datasets.md`](validation_datasets.md).

## Pipeline

```
scRNA-seq ──► Preprocessing ──► OT Trajectory ──► Perturbation Integration
              (QC, HVG, PCA,    (moscot             (VS = OT weight
               UMAP, clustering)  TemporalProblem)     × effect size
                                                      + permutation test)
```

| Step | Script (Kidney) | Script (Brain) | Output |
|------|----------------|----------------|--------|
| 1. Preprocessing | `01_preprocessing.py` | `brain/01_preprocessing_brain.py` | `adata_preprocessed.h5ad`, UMAP/QC figures |
| 2. OT Trajectory | `02_moscot_transport.py` | `brain/02_moscot_transport_brain.py` | `adata_with_transport.h5ad`, transition matrices |
| 3. Perturbation | `03_rsa_integration.py` | `brain/03_perturbation_integration_brain.py` | `vulnerability_scores.csv`, VS figures |

## Repository Structure

```
├── scripts/
│   ├── 01_preprocessing.py                        # Kidney preprocessing
│   ├── 02_moscot_transport.py                     # Kidney OT trajectory inference
│   ├── 03_rsa_integration.py                      # Kidney RSA + vulnerability scoring
│   └── brain/
│       ├── 01_preprocessing_brain.py              # Brain preprocessing
│       ├── 02_moscot_transport_brain.py           # Brain OT trajectory inference
│       └── 03_perturbation_integration_brain.py   # Brain CRISPRi + vulnerability scoring
├── validation_datasets.md      # Detailed analysis process and results per dataset
└── README.md
```

## Data

### Included in this repository
- **Perturbation source data:** `mmc2.xlsx` (kidney RSA scores), `ST5_CRISPRi_enrichment.xls` (brain CRISPRi LOR)
- **OT transition matrices:** CSV files for all computed transitions (kidney: 3, brain: 11)
- **Vulnerability scores:** Final VS with p-values for both datasets

### Not included (download required for Steps 1–2)

| File | Size | Source | Place in |
|------|------|--------|----------|
| Kidney H5 files (×7) | ~120 MB total | [GEO: GSE181595](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181595) | `data/raw/` |
| `RNA_data.h5ad` | ~920 MB | [Zenodo: 10.5281/zenodo.5242913](https://doi.org/10.5281/zenodo.5242913) | `data/fleck2022/raw/` |

> **Note:** Steps 1–2 (preprocessing + OT computation) require the raw H5/h5ad files above. Step 3 (perturbation integration) can run independently using the included CSV transition matrices and supplementary tables.

## Requirements

```
python >= 3.10
scanpy == 1.12
moscot == 0.5.0
anndata
matplotlib
seaborn
numpy
pandas
openpyxl
```

## Usage

```bash
# Full pipeline (requires raw data download)
python scripts/01_preprocessing.py
python scripts/02_moscot_transport.py
python scripts/03_rsa_integration.py

# Brain pipeline
python scripts/brain/01_preprocessing_brain.py
python scripts/brain/02_moscot_transport_brain.py
python scripts/brain/03_perturbation_integration_brain.py
```

## Key Results

| Dataset | Most Vulnerable Transition | Permutation p | Temporal Resolution |
|---------|---------------------------|---------------|-------------------|
| Brain (11 timepoints) | **Excitatory neurogenesis** | **0.0005** | Sufficient |
| Kidney (3 timepoints) | Stromal maturation | < 0.0001 | Limited — near-identity matrices |

The single biggest takeaway: **temporal resolution governs OT quality**. Same pipeline, same parameters — but 3 vs 11 timepoints yields qualitatively different results.

## Known Limitations

1. **Heuristic integration:** VS = OT_weight x effect_size is a post-hoc product, not a joint probabilistic model
2. **Rule-based mapping:** perturbation-to-transition assignment relies on prior knowledge, not learned from data
3. **Population-level readout:** CRISPR screens give bulk effect sizes — per-cell resolution (CROP-seq) is needed for true trajectory-resolved analysis
4. **Temporal resolution matters more than anything else:** 3 timepoints (kidney) produce near-identity matrices that limit the entire downstream analysis

These are not minor caveats — they define what this framework can and cannot claim. The natural next step is CROP-seq integration, where per-cell perturbation identity lives on the trajectory directly.

## References

- Ungricht R, et al. *Genome-wide screening in human kidney organoids identifies developmental and disease-related aspects of nephrogenesis.* Cell Stem Cell (2022).
- Fleck JS, et al. *Inferring and perturbing cell fate regulomes in human brain organoids.* Nature (2022).
- Klein D, et al. *Mapping cells through time and space with moscot.* Nature (2025).
- Schiebinger G, et al. *Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming.* Cell (2019).

## License

Academic coursework. Not intended for clinical or diagnostic use.
