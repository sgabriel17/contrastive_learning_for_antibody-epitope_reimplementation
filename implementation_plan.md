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

| Component | Holt et al. (AbLang-RBD) | AntiBERTy (ours) | ESM-2 (ours) |
|---|---|---|---|
| **Encoder** | AbLang-Heavy + AbLang-Light (dual-stream, 768D each) | AntiBERTy (single encoder, **512D**, heavy-only) | ESM-2-650M (single encoder, **1280D**, H+L concatenated) |
| **Input** | Heavy + Light chains (separate encoders) | Heavy chain only | Both chains concatenated with `<cls>` separator tokens |
| **Pooling** | Mean-pool final hidden states per chain | Mean-pool final hidden states | Mean-pool over all non-special positions |
| **MLP head** | 6-layer MLP, 1536D→1536D (768+768 input) | 6-layer MLP, **512D→512D** | 6-layer MLP, **1280D→1280D** |
| **Fine-tuning** | QLoRA (r=16, α=32, dropout=0.3) on query+value | Identical | Identical |
| **Loss** | Supervised contrastive / NT-Xent (Khosla et al.) | Identical | Identical |
| **Optimizer** | AdamW, lr=1e-5 | Identical | Identical |
| **Batch size** | 256 | Identical | Identical (may need gradient accumulation on T4) |
| **Epochs** | 400 (best checkpoint at epoch 280) | ~400 (to be tuned) | ~400 (to be tuned) |
| **Training data** | 3,093 Bloom Lab DMS sequences, 12 epitopes | Identical | Identical |
| **Evaluation** | Per-epitope weighted AUROC, avg. precision, F1, t-SNE | Identical | Identical |

### AntiBERTy — Heavy Chain Only (512D)
Holt et al. evaluated AntiBERTy as a frozen baseline using **heavy chain only** with mean-pooling. We follow the same approach for fine-tuning to ensure a clean comparison. AntiBERTy's hidden dimension is **512D** (not 768D like AbLang), so the 6-layer MLP head operates at 512D throughout. While AntiBERTy was trained on both heavy and light chains from OAS, using it for paired sequences would be a novel and unvalidated design choice; heavy-only keeps our experiment controlled.

### ESM-2 — Both Chains Concatenated (1280D)
Following Holt et al.'s own benchmarking setup (adapted from Burbach & Briney / BALM), heavy and light chains are fed **simultaneously as a single input** to ESM-2-650M, separated by two `<cls>` tokens: `<cls> H_seq <cls> <cls> L_seq <eos>`. Mean-pooling over all non-special-token positions produces a single **1280D** embedding, which is fed into a 6-layer MLP (1280D→1280D). This matches how ESM-2 was benchmarked in the paper and avoids the dual-stream architecture that AbLang requires (since ESM-2 is a single general-purpose protein encoder, not an antibody-specific H/L pair).

---

## Datasets

### Training Set — Bloom Lab DMS Data
- **Source:** Cao et al. (2022, 2023) via [Bloom Lab GitHub](https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps)
- **Size:** 3,093 SARS-CoV-2 RBD antibodies (filtered from 3,195 for index-strain binding)
- **Labels:** 12 discrete epitope bins derived from deep mutational scanning (DMS) escape profiles
- **What DMS is:** A high-throughput wet-lab technique that systematically introduces every possible single amino acid mutation to the RBD and measures how much each mutation disrupts antibody binding. The resulting "escape map" defines the antibody's functional epitope. Antibodies with overlapping escape profiles are assigned to the same bin.
- **Split:** 80% train / 10% validation / 10% test, partitioned by clonal clusters defined as shared heavy V-gene AND >70% CDRH3 amino acid identity. No antibody in the same cluster appears in both train and test. (Note: the 65% CDRH3 cutoff mentioned elsewhere in the paper applies to the AbLang-PDB/SAbDab dataset, not the RBD dataset we use.)
- **Pre-computed splits:** The existing `ablang_model/train/rbd_dataset.pd` file already contains a `DATASET` column with `TRAIN`/`TEST`/`VAL` labels from Holt et al.'s pipeline. We use these directly to ensure our splits match theirs exactly.
- **Balanced oversampling:** During training, minority epitope classes are upsampled to match the count of the largest class (via the `oversample_epitope` function in Holt et al.'s code). This ensures the contrastive loss sees balanced representations of all 12 epitopes per epoch.

### Independent Test Set — CoV-AbDab
- **Source:** [Raybould et al. (2021)](https://doi.org/10.1093/bioinformatics/btaa739) — Coronavirus Antibody Database
- **Size:** 237 RBD-specific antibodies with crystal structure data from the PDB
- **Labels:** Continuous buried surface area (BSA) overlap scores (Å²) computed from crystal structures
- **Purpose:** Evaluates whether learned embeddings correlate with *structural* (not just functional) epitope overlap. A cosine similarity threshold of 0.85 distinguishes high BSA overlap (>750 Å²) with 97% accuracy in the AbLang-RBD baseline. A *nice goal* would be if our finetuned models are able to take a pair of antibodies that are quite different sequence-wise but bind to the same epitope and embed them similarly. Similarly, take a pair of antibodies that are quite similar in sequence but bind different epitopes and see if our models embed them differently (as they should).

---

## Model Architectures (Per Encoder)

### AntiBERTy Pipeline (heavy chain only → 512D)
```
Input: heavy_seq (amino acid string)

  → Space-separate AAs ("E V Q L ...")
  → Tokenize (AntiBERTy tokenizer)
  → AntiBERTy encoder (12 transformer blocks, 512D hidden) + QLoRA adapters
  → Mean-pool final hidden states over non-special-token positions → 512D

  → 6-layer MLP (Mixer): 512D → ReLU → 512D → ReLU → 512D → ReLU → 512D → ReLU → 512D → ReLU → 512D
  → L2-normalize → 512D unified antibody embedding
```

### ESM-2 Pipeline (both chains concatenated → 1280D)
```
Input: (heavy_seq, light_seq)

  → Concatenate with CLS separators: "<cls> H_seq <cls> <cls> L_seq <eos>"
  → Tokenize (ESM-2 tokenizer)
  → ESM-2-650M encoder (33 transformer blocks, 1280D hidden) + QLoRA adapters
  → Mean-pool final hidden states over non-special-token positions → 1280D

  → 6-layer MLP (Mixer): 1280D → ReLU → 1280D → ReLU → 1280D → ReLU → 1280D → ReLU → 1280D → ReLU → 1280D
  → L2-normalize → 1280D unified antibody embedding
```

### Shared Training Configuration
```
Loss:       Supervised contrastive (NT-Xent, τ=0.5, batch_size=256)
Frozen:     All pretrained transformer weights
Trainable:  QLoRA adapter matrices (r=16, α=32, dropout=0.3, target_modules=["query", "value"])
            + 6-layer MLP head (all parameters)
Optimizer:  AdamW, lr=1e-5
Epochs:     400 (select best checkpoint by validation AUROC)
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

All metrics match Holt et al. exactly for apples-to-apples comparison.

### Per-Epitope Weighted AUROC (Primary Metric)
The paper does **not** use a simple global binary AUROC. Instead, it computes a **per-epitope weighted AUROC** (see `get_cross_dataset_weighted_rocauc` in the upstream `analysis.py`):
1. For each of the 12 epitope classes, compute class-specific TPR/FPR curves independently
2. Weight each class's curve by its proportion of positive pairs
3. Average the weighted TPR/FPR curves across classes
4. Compute AUC of the combined curve

This matters because a global binary AUROC (as used in our frozen baseline notebook) produces inflated numbers (e.g., 0.812 instead of the paper's 0.73 for test-vs-test). All final reported numbers must use the per-epitope weighted method.

### Two Evaluation Modes
- **Train-vs-test:** Embed all training antibodies and all test antibodies; compute pairwise cosine similarities between train and test sets. This is the paper's primary metric (AbLang-RBD AUROC = 0.84).
- **Test-vs-test:** Compute pairwise cosine similarities within the test set only. This is harder and more realistic (AbLang-RBD AUROC = 0.73).

### Full Metric Suite

| Metric | Description |
|---|---|
| **Per-epitope weighted AUROC** | Weighted average of per-class ROC curves (see above) |
| **Balanced accuracy** | Mean per-class accuracy across all 12 epitope bins, at optimal threshold from validation |
| **Average precision** | Area under precision-recall curve |
| **F1 score** | At the cosine similarity threshold that maximizes balanced accuracy on validation |
| **Spearman ρ** | Rank correlation between cosine similarity and BSA overlap (CoV-AbDab structural test) |
| **t-SNE** | 2D visualization of embeddings colored by epitope bin; k-means accuracy (k=12) reported |

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

## What's Done and What's Left

### Completed (Milestone)
- [x] Frozen baseline notebook (`notebooks/01_frozen_baseline.ipynb`): loads pre-computed AbLangRBD1 embeddings + frozen AntiBERTy/ESM-2, computes binary AUROC and t-SNE
- [x] Baseline results: AbLangRBD1=0.812 (binary AUROC, test-vs-test), AntiBERTy(frozen)=0.546, ESM-2(frozen)=0.547

### To Build (Training Pipeline — `src/`)
- [ ] `src/data_loader.py` — load `rbd_dataset.pd`, balanced oversampling, tokenization for both encoders
- [ ] `src/models.py` — `AntiBERTyContrastive` (512D, heavy-only) and `ESM2Contrastive` (1280D, H+L concat) with QLoRA + 6-layer MLP
- [ ] `src/loss.py` — port `ContrastiveLoss` and `ContrastiveTrainTestLoss` from upstream code
- [ ] `src/train.py` — training loop with AdamW, validation AUROC, mixed precision, **periodic checkpoints + resume** (see below)
- [ ] `src/evaluate.py` — per-epitope weighted AUROC (matching paper), avg precision, F1, t-SNE, both eval modes
- [ ] Training notebook for Colab (`notebooks/02_train.ipynb`) — imports from `src/`, runs on A100/G4; **mount Drive or sync `output_dir`** so checkpoints survive disconnects

#### Checkpointing (Colab / interrupted runs)
Training must save **resumable** state regularly so a disconnect does not lose work:
- Every *N* epochs (e.g. 5 or 10): save `checkpoint_epoch_{e}.pt` containing model state_dict, optimizer state_dict, epoch number, random RNG states (optional), and training metrics JSON.
- Always overwrite `checkpoint_latest.pt` with the same contents (quick resume).
- Optional: keep `checkpoint_best.pt` when validation AUROC improves.
- CLI: `--resume path/to/checkpoint_latest.pt` to continue from the last completed epoch.
- In Colab: set `--output_dir` to Google Drive or copy `output_dir` to Drive at the end of each session.

### Nice-to-Have (If Time Permits)
- [ ] CoV-AbDab structural validation (Spearman ρ with BSA overlap)
- [ ] Cosine similarity distribution plots (same-epitope vs. different-epitope)
- [ ] Update `01_frozen_baseline.ipynb` to use per-epitope weighted AUROC for corrected baseline numbers

---

## Project Scope and Limitations

- **Scope:** We apply Holt et al.'s QLoRA + contrastive fine-tuning framework to two alternative encoders (AntiBERTy and ESM-2). Hyperparameters, dataset, loss function, and evaluation protocol are kept identical to the paper. The only variables are the encoder backbone and its associated architectural adaptations (MLP input dimension, tokenization).
- **We are NOT implementing AbLang-PDB** (the generalized cross-antigen model). Scope is limited to SARS-CoV-2 RBD antibodies only.
- **Compute:** Google Colab Pro (A100 40GB or RTX 6000/G4). Training AbLang-RBD took ~5h on an A6000; AntiBERTy (~26M params) should be faster, ESM-2-650M may need gradient accumulation for batch size 256 on smaller GPUs.
- **AntiBERTy uses heavy chain only** (matching Holt et al.'s frozen baseline setup). This avoids introducing an uncontrolled variable (paired-input for a model not trained on paired data).

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

### Model Code (`src/models.py`)
- `AntiBERTyContrastive`: loads AntiBERTy, injects QLoRA, adds 6-layer MLP (512D). Interface: `forward(h_input_ids, h_attention_mask) -> embedding` (heavy chain only).
- `ESM2Contrastive`: loads ESM-2-650M, injects QLoRA, adds 6-layer MLP (1280D). Interface: `forward(input_ids, attention_mask) -> embedding` (concatenated H+L input).
- The 6-layer MLP (`Mixer`) keeps input dimension = output dimension, with ReLU between all layers **except** the final layer. Output is L2-normalized.

### QLoRA Configuration
- `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.3, target_modules=["query", "value"], bias="none", task_type="FEATURE_EXTRACTION")`
- The `target_modules` strings must match each encoder's actual module names:
  - **AntiBERTy (BERT-based):** likely `"query"`, `"value"` in `BertSelfAttention`
  - **ESM-2:** likely `"q_proj"`, `"v_proj"` in `ESMSelfAttention` — verify with `model.named_modules()`
- Training freezes all pretrained weights; only QLoRA adapters + MLP head are trainable.
- Confirm with `sum(p.numel() for p in model.parameters() if p.requires_grad)` before starting.

### Loss (`src/loss.py`)
- The contrastive loss expects: embeddings `[batch_size, embed_dim]` + labels `[batch_size]` (integer epitope indices). After filtering to the 12 main bins, labels are 0–11; the raw pickle has labels **0–15** if held-out epitopes are included.
- Port directly from `ablang_model/train/models.py` classes `ContrastiveLoss` (training) and `ContrastiveTrainTestLoss` (evaluation).
- Temperature τ = 0.5.

### Data (`src/data_loader.py`)
- Source: `ablang_model/train/rbd_dataset.pd` — use existing `DATASET` column for splits.
- Balanced oversampling of TRAIN set (minority epitopes upsampled to max class count).
- AntiBERTy tokenizer: space-separate AAs before tokenizing (e.g., `"E V Q L ..."`).
- ESM-2 tokenizer: concatenate chains with `<cls>` separators, tokenizer handles the rest.

#### `rbd_dataset.pd` (inspected, 3,143 rows × 25 columns)
Key columns: `HC_AA`, `LC_AA`, `EPITOPE`, `EPITOPE_LABELS` (integers **0–15**), `DATASET`, `CLONOTYPE`, `PREPARED_HC_SEQ` / `PREPARED_LC_SEQ`, `EMBEDDING` (pretrained AbLang vector per row), `SYNTHETIC`, `CLASS`, V/D/J fields, etc.
- **16 distinct `EPITOPE` strings:** the 12 RBD DMS bins (A, B, C, D1, D2, E1, E2.1, E2.2, E3, F1, F2, F3) **plus** four held-out variants (`A-BA1`, `B-BA1`, `D-BA1`, `F3-BA1`) as in the upstream `get_run_specifics.py` `HELD_OUT_EPITOPES`. For the **12-class Holt benchmark**, filter rows to the main epitopes (or to `EPITOPE_LABELS` in 0–11) unless explicitly running a held-out experiment.
- **`DATASET` is mixed case:** `TRAIN` / `TEST` / `VAL` (majority) **and** a small `train` / `test` subset (102 rows total — likely additional held-out or duplicate labeling). The data loader should filter using **one convention** (e.g. `DATASET.isin(['TRAIN','TEST','VAL'])` in uppercase) and document the choice.
- **Train/test sizes (uppercase only):** TRAIN 2,465; TEST 307; VAL 269 (matches the frozen baseline notebook’s TEST=307).

### Evaluation (`src/evaluate.py`)
- Evaluation is always done in two modes: (1) **train-vs-test** (compare test embeddings to train embeddings pairwise) and (2) **test-vs-test** (compare held-out antibodies to each other).
- Use **per-epitope weighted AUROC** (port from `ablang_model/train/analysis.py` `get_cross_dataset_weighted_rocauc`), NOT sklearn's `roc_auc_score`.

### Reference Code
- The upstream Holt et al. training code lives in `ablang_model/train/` and serves as the ground-truth reference for all architectural and training decisions. Key files: `models.py` (model + loss), `run_simclr_250129.py` (training loop), `data_handling.py` (data loading + oversampling), `analysis.py` (evaluation + plotting), `get_run_specifics.py` (hyperparameters).
