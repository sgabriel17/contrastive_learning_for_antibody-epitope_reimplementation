# Contrastive Fine-Tuning of Antibody Language Models for Epitope-Aware Embeddings

**Course:** 6.8711 / 18.S997 — Deep Learning in the Life Sciences (Spring 2026)
**Team:** Gabriel Sanchez, Gabriela Erin Mariangel, Natalie Barnouw
**Based on:** [Holt et al. (2026), *Patterns*](https://doi.org/10.1016/j.patter.2025.101419)
**Upstream code:** [IGlab-VUMC/AbLangRBD1](https://github.com/IGlab-VUMC/AbLangRBD1) (adapted, not forked)

## Overview

Pretrained antibody and protein language models cluster sequences by germline similarity, not by functional epitope. Holt et al. showed that supervised contrastive QLoRA fine-tuning of AbLang yields **AbLang-RBD** embeddings that group SARS-CoV-2 RBD antibodies by epitope.

This repo asks whether the same pipeline can improve **AntiBERTy** (heavy chain only, 512D) and **ESM-2-650M** (heavy + light concatenated, 1280D). We keep the dataset, loss, optimizer, and evaluation protocol identical to Holt et al.; only the encoder backbone changes.

## Repository layout

```
.
├── ablang_model/train/       # Holt et al. reference code + rbd_dataset.pd
├── src/
│   ├── data_loader.py        # Splits, oversampling, tokenization
│   ├── models.py             # AntiBERTy / ESM-2 wrappers + QLoRA + 6-layer MLP
│   ├── loss.py               # Supervised contrastive (NT-Xent) loss
│   ├── train.py              # Training loop with checkpointing
│   └── evaluate.py           # Metrics and figure generation
├── notebooks/
│   ├── frozen_baseline.ipynb # Frozen AntiBERTy, ESM-2, and AbLang-RBD baselines
│   └── train.ipynb           # Colab launcher for src/train.py
├── results/                  # Saved metrics JSON and figures
├── preprocess_dms_escape_data/  # Optional DMS preprocessing scripts
└── requirements.txt
```

**Data:** Training uses Holt's pre-split pickle at `ablang_model/train/rbd_dataset.pd` (3,041 antibodies across 12 DMS epitope bins; TRAIN/VAL/TEST splits).

## Setup

```bash
git clone https://github.com/sgabriel17/contrastive_learning_for_antibody-epitope_reimplementation.git
cd contrastive_learning_for_antibody-epitope_reimplementation

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`bitsandbytes` (4-bit QLoRA) requires Linux/CUDA — use Google Colab for training. On macOS, install everything else and run evaluation locally.

**Model weights** (downloaded automatically on first use):

| Model | Source |
| --- | --- |
| AntiBERTy | `pip install antiberty` |
| ESM-2-650M | `facebook/esm2_t33_650M_UR50D` (HuggingFace) |
| AbLang-RBD (baseline) | `clint-holt/AbLangRBD1` |

## Training

```bash
python src/train.py \
  --encoder esm2 \
  --output_dir results/esm2_results \
  --epochs 400 \
  --batch_size 256 \
  --grad_accum_steps 1 \
  --lr 1e-5 \
  --temperature 0.5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.3 \
  --checkpoint_every 5 \
  --eval_every 1
```

Use `--encoder antiberty` for AntiBERTy. Set `--resume path/to/checkpoint_latest.pt` to continue an interrupted run.

Checkpoints written to `--output_dir`:

- `checkpoint_best.pt` — best validation AUROC
- `checkpoint_latest.pt` — most recent epoch (for resume)
- `checkpoint_epoch_NNN.pt` — periodic snapshots
- `metrics.json` — per-epoch loss and validation metrics
- `run_config.json` — CLI arguments

For Colab, open `notebooks/train.ipynb` (mounts Drive, clones repo, launches training).

## Evaluation

```bash
python src/evaluate.py \
  --checkpoint results/esm2_results/checkpoint_best.pt \
  --encoder esm2 \
  --plot \
  --output_dir results/esm2_results
```

Writes `final_eval.json` and, with `--plot`, saves:

- `tsne_test.png`, `tsne_train_test.png`
- `cosine_distributions.png`

## Metrics

All reported numbers follow Holt et al.'s protocol (implemented in `src/evaluate.py`):

| Metric | Notes |
| --- | --- |
| **AUROC** | Per-epitope weighted AUROC (`weighted_auroc` in JSON) |
| **Precision** | Average precision (`average_precision`) |
| **F1** | At a fixed cosine-similarity threshold |
| **Accuracy** | Balanced accuracy (`balanced_accuracy`) |

Thresholds are chosen on the validation set (train-vs-val pairs) and applied unchanged to test evaluation.

**Evaluation modes** (both reported in `final_eval.json`):

- **Train vs. test** — pairwise cosine similarity between train and held-out test antibodies
- **Test vs. test** — pairwise similarity within the test set only (harder, primary comparison to Holt et al.)

**Figures:** cosine similarity distributions (same- vs. different-epitope pairs) and t-SNE plots colored by epitope bin.

## Results

Pre-computed outputs live under `results/`:

| Directory | Contents |
| --- | --- |
| `frozen_fixed/` | Frozen AntiBERTy, ESM-2, and AbLang-RBD baselines |
| `antiberty_results/` | Fine-tuned AntiBERTy |
| `esm2_results/` | Fine-tuned ESM-2 |
| `comparison_table_FINAL.png` | Summary comparison across models |

See each directory's `final_eval.json` for numeric metrics.

## Training configuration

Shared across both encoders (matching Holt et al.):

- **Loss:** supervised contrastive / NT-Xent, τ = 0.5, batch size 256
- **Fine-tuning:** QLoRA (r = 16, α = 32, dropout = 0.3) on query + value projections
- **Head:** 6-layer MLP (dim-preserving) + L2 normalization
- **Optimizer:** AdamW, lr = 1e-5
- **Training data:** balanced oversampling of minority epitope classes

**Encoder-specific input:**

- **AntiBERTy** — heavy chain only, space-separated amino acids, mean-pooled → 512D
- **ESM-2** — `<cls> H_seq <cls> <cls> L_seq <eos>`, mean-pooled → 1280D

## References

- Holt et al. (2026). Contrastive fine-tuning of antibody language models. *Patterns*.
- Khosla et al. (2020). Supervised contrastive learning.
- Hu et al. (2021). LoRA / QLoRA.
- Ruffolo et al. (2021). AntiBERTy.
- Lin et al. (2023). ESM-2.
- Cao et al. (2022, 2023). Bloom Lab SARS-CoV-2 RBD DMS escape data.
