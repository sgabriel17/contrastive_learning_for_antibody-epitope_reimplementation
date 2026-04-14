# Contrastive Fine-Tuning of Antibody Language Models for Epitope-Aware Embeddings

**Course:** 18.S997 / 6.8711 — Machine Learning in Computational Biology  
**Team:** Gabriel Sanchez, Gabriela Erin Mariangel, Natalie Barnouw  
**Based on:** Holt et al. (2026), *Patterns* — [DOI: 10.1016/j.patter.2025.101419](https://doi.org/10.1016/j.patter.2025.101419)  
**Upstream code:** [IGlab-VUMC/AbLangRBD1](https://github.com/IGlab-VUMC/AbLangRBD1) (adapted, not forked)

---

## Project Overview

Epitope mapping — identifying which region of an antigen a given antibody binds — is a critical bottleneck in vaccine development and therapeutic antibody discovery. While X-ray crystallography remains the gold standard, AI-based embedding approaches offer a faster and more generalizable alternative.

Pretrained protein language models (PLMs) such as AntiBERTy and ESM-2 naturally cluster antibodies by **sequence similarity and germline gene usage**, not by functional epitope identity. This germline bias means that structurally diverse antibodies targeting the same epitope are far apart in embedding space, while sequence-similar antibodies targeting different epitopes can appear falsely close.

**Holt et al. (2026)** demonstrated that supervised contrastive fine-tuning of the antibody-specific language model AbLang into **AbLang-RBD** yields embeddings that correctly group SARS-CoV-2 RBD antibodies by epitope — achieving 74.4% balanced accuracy and an AUROC of 0.84 on held-out test sequences. Crucially, they evaluated AntiBERTy and ESM-2 only as **frozen, pretrained baselines** (no fine-tuning), where both performed barely above random (AUROC ≈ 0.57 and 0.56 respectively).

**Our project** asks: *can the same QLoRA-based supervised contrastive framework rescue AntiBERTy and ESM-2's performance to the level of AbLang-RBD?* We apply the identical training pipeline to these two alternative encoders and conduct a direct, controlled comparison.

---

## Scientific Background

### What Is Epitope Mapping?
An **epitope** is the specific region of an antigen (e.g., the SARS-CoV-2 receptor-binding domain, RBD) that an antibody physically contacts. Two antibodies are said to have "overlapping epitopes" if they compete for the same binding site. Knowing this is critical for:
- Selecting synergistic antibody combinations for therapeutics
- Understanding immune response breadth across variants
- Rapid down-selection of antibody candidates

### What Is Contrastive Learning?
Supervised contrastive learning (Khosla et al., 2020) trains a model to pull together embeddings from the **same class** (same epitope bin) while pushing apart embeddings from **different classes** (different epitope bins). Unlike standard classification, it works over all positive pairs in a batch simultaneously, making it especially powerful for learning fine-grained similarity structure.

The loss function (NT-Xent with multi-positive support) is:

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{|P_i|} \sum_{j \in P_i} -\log \frac{\exp(S_{ij})}{\sum_{k 
eq i} \exp(S_{ik})}$$

Where:
- $B = 256$ (batch size)
- $S_{ij} = z_i^T z_j / \tau$ (scaled cosine similarity, $\tau = 0.5$)
- $P_i = \{j : y_j = y_i, j 
eq i\}$ (set of same-epitope antibodies in the batch)

### What Is QLoRA?
QLoRA (Quantized Low-Rank Adaptation, Hu et al., 2021) is a parameter-efficient fine-tuning method that:
1. **Freezes** the original pretrained transformer weights
2. Injects small **low-rank adapter matrices** (rank $r = 16$, scaling factor $\alpha = 32$) into the attention layers
3. Trains only those adapters (and the MLP head) — reducing trainable parameters by ~99% vs. full fine-tuning

This makes it feasible to fine-tune large models like ESM-2 (650M parameters) on modest GPU hardware.

---

## What We Are Building vs. What Holt et al. Built

| Component | Holt et al. (AbLang-RBD) | This Project |
|---|---|---|
| **Heavy chain encoder** | AbLang-Heavy (RoBERTa, 12-layer, 768D) | AntiBERTy (12-layer) / ESM-2-650M |
| **Light chain encoder** | AbLang-Light (RoBERTa, 12-layer, 768D) | AntiBERTy / ESM-2-650M (or heavy-only for AntiBERTy) |
| **Pooling** | Mean-pool final hidden states | Identical |
| **Cross-chain head** | 6-layer MLP → 1,536D unified embedding, ReLU activations (no final activation) | Identical |
| **Fine-tuning method** | QLoRA (r=16, α=32, dropout=0.3) | Identical |
| **Loss function** | Supervised contrastive / NT-Xent (Khosla et al.) | Identical |
| **Optimizer** | AdamW, lr=1e-5 | Identical |
| **Batch size** | 256 | Identical |
| **Training epochs** | 400 (best checkpoint at epoch 280) | ~400 (to be tuned) |
| **Training data** | 3,093 Bloom Lab DMS sequences, 12 epitopes | Identical |
| **Evaluation** | Balanced accuracy, AUROC, avg. precision, F1, Spearman ρ, t-SNE | Identical |

### AntiBERTy Chain Handling — Important Design Decision
AntiBERTy was trained on both heavy and light chains (from the Observed Antibody Space database), but Holt et al. used only heavy-chain averaging in their frozen baseline. Our project **feeds both chains** through a single AntiBERTy encoder and applies the same 6-layer MLP head. This is a novel usage relative to Holt et al. and constitutes an additional experimental axis: does light-chain information improve contrastive fine-tuning performance?

### ESM-2 Chain Handling
Following Holt et al.'s own benchmarking setup, heavy and light chains are fed **simultaneously, separated by two `[CLS]` tokens**, into the ESM-2-650M model. Mean-pooling is applied over the final hidden states.

---

## Datasets

### Training Set — Bloom Lab DMS Data
- **Source:** Cao et al. (2022, 2023) via [Bloom Lab GitHub](https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps)
- **Size:** 3,093 SARS-CoV-2 RBD antibodies (filtered from 3,195 for index-strain binding)
- **Labels:** 12 discrete epitope bins derived from deep mutational scanning (DMS) escape profiles
- **What DMS is:** A high-throughput wet-lab technique that systematically introduces every possible single amino acid mutation to the RBD and measures how much each mutation disrupts antibody binding. The resulting "escape map" defines the antibody's functional epitope. Antibodies with overlapping escape profiles are assigned to the same bin.
- **Split:** 80% train / 10% validation / 10% test, partitioned by heavy V-gene usage + CDRH3 cluster (≥70% CDRH3 identity cutoff of 65%) — ensures no antibody in the same clonal cluster appears in both train and test

### Independent Test Set — CoV-AbDab
- **Source:** [Raybould et al. (2021)](https://doi.org/10.1093/bioinformatics/btaa739) — Coronavirus Antibody Database
- **Size:** 237 RBD-specific antibodies with crystal structure data from the PDB
- **Labels:** Continuous buried surface area (BSA) overlap scores (Å²) computed from crystal structures
- **Purpose:** Evaluates whether learned embeddings correlate with *structural* (not just functional) epitope overlap. A cosine similarity threshold of 0.85 distinguishes high BSA overlap (>750 Å²) with 97% accuracy in the AbLang-RBD baseline.

---

## Model Architecture (Detailed)

```
Input: paired (heavy_seq, light_seq) antibody sequences

Heavy Chain:
  → Tokenize (HuggingFace AutoTokenizer)
  → Encoder (AntiBERTy or ESM-2) + QLoRA adapters
  → Mean-pool final hidden states over non-masked positions
  → 768D heavy embedding  [for AbLang/AntiBERTy] or 1280D [for ESM-2-650M]

Light Chain: (same encoder, separate pass)
  → 768D / 1280D light embedding

Cross-chain MLP head:
  → Concatenate [heavy_emb; light_emb] → 1536D (or 2560D for ESM-2)
  → FC → ReLU → FC → ReLU → FC → ReLU → FC → ReLU → FC → ReLU → FC
  → L2-normalize → 1536D unified antibody embedding

Training objective:
  → Supervised contrastive loss (NT-Xent, τ=0.5, batch_size=256)
  → Frozen: all pretrained transformer weights
  → Trainable: QLoRA adapter matrices (r=16, α=32, dropout=0.3) + 6 MLP layers
```

---

## Key Results from Holt et al. (Our Baselines to Beat)

| Model | Condition | AUROC | Avg. Precision | F1 | Balanced Acc. |
|---|---|---|---|---|---|
| AbLang (pretrained, frozen) | Train vs. Test | 0.57 | 0.18 | 0.21 | 56.0% |
| **AbLang-RBD (fine-tuned)** | **Train vs. Test** | **0.84** | **0.64** | **0.59** | **82.7%** |
| AbLang-RBD (fine-tuned) | Test vs. Test | 0.73 | 0.39 | 0.39 | 74.4% |
| AntiBERTy (frozen baseline) | Train vs. Test | 0.57 | 0.17 | 0.21 | — |
| ESM-2 (frozen baseline) | Train vs. Test | 0.56 | 0.18 | 0.20 | — |

Our goal is to show that **QLoRA fine-tuning of AntiBERTy and/or ESM-2** brings their AUROC and avg. precision significantly above the frozen baseline and toward AbLang-RBD levels.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                        # Downloaded source files — NOT committed to git
│   │   ├── bloom_dms_escape.csv    # Bloom Lab DMS escape data (all mutations)
│   │   ├── dms_epitope_bins.csv    # Pre-computed 12-bin epitope labels
│   │   └── covabdab_rbd.csv        # CoV-AbDab RBD-specific subset with BSA overlap
│   └── processed/                  # Tokenized / split datasets
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
│
├── src/
│   ├── data_loader.py              # Dataset class, collate fn, train/val/test splits
│   ├── models.py                   # Encoder wrappers (AntiBERTy, ESM-2), MLP head, QLoRA injection
│   ├── loss.py                     # Supervised contrastive loss (NT-Xent, multi-positive)
│   ├── train.py                    # Training loop: AdamW, checkpoint saving, validation
│   └── evaluate.py                 # AUROC, avg. precision, F1, Spearman ρ, t-SNE/UMAP plots
│
├── notebooks/
│   ├── 01_frozen_baseline.ipynb    # Extract embeddings (no fine-tuning), UMAP visualization
│   ├── 02_training_curves.ipynb    # Plot loss and AUROC across epochs
│   └── 03_results_analysis.ipynb  # BSA correlation, cosine similarity distributions
│
├── results/
│   └── figures/                    # Saved .png outputs (UMAP plots, training curves, etc.)
│
├── checkpoints/                    # Saved model weights — NOT committed to git
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup and Installation

### 1. Clone this repository
```bash
git clone https://github.com/sgabriel17/contrastive_learning_for_antibody-epitope_reimplementation.git
cd contrastive_learning_for_antibody-epitope_reimplementation
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# or: venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the data
```bash
# Bloom Lab DMS escape data (Cao et al. 2022, 2023)
wget https://media.githubusercontent.com/media/jbloomlab/SARS2_RBD_Ab_escape_maps/refs/heads/main/processed_data/escape_data.csv      -O data/raw/bloom_dms_escape.csv

# CoV-AbDab — download from https://opig.stats.ox.ac.uk/webapps/covabdab/
# Filter for RBD-specific antibodies with PDB structural data
```

---

## Dependencies (`requirements.txt`)

```
torch>=2.0
transformers>=4.38
peft>=0.10              # QLoRA / LoRA via HuggingFace PEFT
bitsandbytes>=0.42      # 4-bit quantization for QLoRA
antiberty               # pip install antiberty
fair-esm                # ESM-2 (Meta): pip install fair-esm
scikit-learn>=1.3
scipy
numpy
pandas
matplotlib
seaborn
umap-learn
biopython
tqdm
```

> **Note on model weights:**
> - **AbLang-RBD** (Holt et al. fine-tuned): `clint-holt/AbLangRBD1` on HuggingFace
> - **AbLang (pretrained):** `qilowoq/AbLang-heavy` and `qilowoq/AbLang-light`
> - **AntiBERTy:** via `pip install antiberty` (Ruffolo et al., 2021)
> - **ESM-2-650M:** `facebook/esm2_t33_650M_UR50D` on HuggingFace

---

## Running Experiments

### Frozen baseline (no fine-tuning)
```bash
python notebooks/01_frozen_baseline.ipynb
# Extracts embeddings from pretrained AntiBERTy / ESM-2
# Generates UMAP colored by epitope bin
# Computes AUROC, avg. precision on DMS test set and CoV-AbDab
```

### Training
```bash
python src/train.py   --encoder antiberty \       # or: esm2, ablang
  --qlora_r 16   --qlora_alpha 32   --qlora_dropout 0.3   --batch_size 256   --lr 1e-5   --epochs 400   --temperature 0.5   --output_dir checkpoints/antiberty_run1
```

### Evaluation
```bash
python src/evaluate.py   --checkpoint checkpoints/antiberty_run1/best.pt   --encoder antiberty   --test_set dms           # or: covabdab
```

---

## Evaluation Metrics

All metrics match Holt et al. exactly for apples-to-apples comparison:

| Metric | Description |
|---|---|
| **Balanced accuracy** | Mean per-class accuracy across all 12 epitope bins |
| **AUROC** | Area under ROC curve for same-epitope vs. different-epitope pair classification |
| **Average precision** | Area under precision-recall curve |
| **F1 score** | At the cosine similarity threshold that maximizes balanced accuracy on validation |
| **Spearman ρ** | Rank correlation between cosine similarity and BSA overlap (CoV-AbDab) |
| **t-SNE / UMAP** | 2D visualization of 1536D embeddings colored by epitope bin; k-means accuracy reported |

Thresholds are determined on the **validation set** and applied to the test set without re-tuning.

---

## Literature

| Reference | Relevance |
|---|---|
| Holt et al. (2026). *Patterns* 7, 101419 | **Primary reference.** Introduces AbLang-RBD via supervised contrastive fine-tuning. Our starting codebase and main comparison target. |
| Khosla et al. (2020). arXiv:2004.11362 | Supervised contrastive learning loss (NT-Xent with multi-positive support) |
| Hu et al. (2021). arXiv:2106.09685 | LoRA / QLoRA — parameter-efficient fine-tuning via low-rank adapters |
| Ruffolo et al. (2021). arXiv:2112.07782 | **AntiBERTy** — antibody LM trained on 558M sequences from OAS; our first alternative encoder |
| Lin et al. (2023). *Science* 379, 1123 | **ESM-2** (650M param) — general protein LM; our second alternative encoder |
| Olsen et al. (2022). *Bioinformatics Advances* 2, vbac046 | **AbLang** — antibody LM trained via masked language modeling; backbone of Holt et al. |
| Cao et al. (2022). *Nature* 608, 593 | Bloom Lab DMS escape data — BA.2.12.1, BA.4, BA.5 |
| Cao et al. (2023). *Nature* 614, 521 | Bloom Lab DMS escape data — Omicron convergent evolution |
| Raybould et al. (2021). *Bioinformatics* 37, 734 | **CoV-AbDab** — Coronavirus Antibody Database; our structural test set |
| Dang et al. (2023). *mAbs* 15 | Epitope mapping method comparison (background motivation) |

---

## Project Scope and Limitations

- **Scope:** We reproduce AbLang-RBD (Holt et al.) and swap only the encoder backbone (AbLang → AntiBERTy, then ESM-2). All other architectural components, hyperparameters, datasets, and evaluation protocols are kept identical.
- **We are NOT implementing AbLang-PDB** (the generalized cross-antigen model). Scope is limited to SARS-CoV-2 RBD antibodies only.
- **Compute constraint:** Training AbLang-RBD took ~5 hours on a single NVIDIA A6000 GPU. ESM-2-650M and AntiBERTy are larger — budget additional time. Specify your compute access (e.g., MIT Satori, Google Colab A100) in experiment logs.
- **AntiBERTy chain handling is a novel axis:** Holt et al. only used heavy-chain averaging for AntiBERTy in their frozen baseline. We feed both chains, which is an additional experimental design choice worth reporting.

---

## Course Deliverables

| Deliverable | Weight | Notes |
|---|---|---|
| Proposal | 10% | Submitted |
| **Milestone** | **20%** | 2-page ICML-format paper; preliminary results required |
| Presentations | 30% | 10 min + 2 min Q&A |
| Paper + Code | 40% | ≤4 pages (excl. references); ICML template; code submitted simultaneously |

**ICML template:** https://github.com/icml-compbio/icml-compbio.github.io/raw/master/2022/icml_wcb_2022.zip

---

## Notes for Cursor / AI Coding Assistants

- All model-related code lives in `src/models.py`. When wrapping a new encoder, implement a class with a `forward(heavy_seq, light_seq) -> embedding` interface.
- The contrastive loss expects a tensor of shape `[batch_size, embed_dim]` and a labels tensor of shape `[batch_size]` (integer epitope bin indices 0–11).
- QLoRA is injected using HuggingFace `peft.get_peft_model()` with `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.3, task_type="FEATURE_EXTRACTION")`.
- The 6-layer MLP head takes the concatenated chain embeddings `[heavy_emb; light_emb]` and outputs an L2-normalized 1536D vector. The final layer has **no activation** before normalization.
- Training freezes all parameters except QLoRA adapters and the MLP head. Confirm this with `sum(p.numel() for p in model.parameters() if p.requires_grad)` before starting a run.
- Evaluation is always done in two modes: (1) **train-vs-test** (embed test antibodies, compare to train set centroids) and (2) **test-vs-test** (compare held-out antibodies to each other).
