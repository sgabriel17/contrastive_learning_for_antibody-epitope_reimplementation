"""Data loading and tokenization for RBD contrastive fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertTokenizer


MAIN_EPITOPES = ["A", "B", "C", "D1", "D2", "E1", "E2.1", "E2.2", "E3", "F1", "F2", "F3"]
DATASET_SPLITS = ("TRAIN", "VAL", "TEST")
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "ablang_model" / "train" / "rbd_dataset.pd"
ESM2_MODEL_ID = "facebook/esm2_t33_650M_UR50D"


@dataclass(frozen=True)
class RBDExample:
    heavy: str
    light: str
    label: int
    epitope: str
    dataset: str


class RBDDataset(Dataset[RBDExample]):
    """Tiny map-style dataset over heavy/light chains and 12-bin epitope labels."""

    def __init__(self, df: pd.DataFrame):
        self.examples = [
            RBDExample(
                heavy=str(row.HC_AA),
                light=str(row.LC_AA),
                label=int(row.EPITOPE_LABELS),
                epitope=str(row.EPITOPE),
                dataset=str(row.DATASET),
            )
            for row in df.itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RBDExample:
        return self.examples[index]


def load_rbd_dataframe(data_path: str | Path = DEFAULT_DATA_PATH, main_epitopes_only: bool = True) -> pd.DataFrame:
    """Load Holt's RBD pickle using uppercase splits and canonical 12-bin labels."""

    df = pd.read_pickle(data_path).copy()
    df = df[df["DATASET"].isin(DATASET_SPLITS)]
    if main_epitopes_only:
        df = df[df["EPITOPE"].isin(MAIN_EPITOPES)]

    df = df.dropna(subset=["HC_AA", "LC_AA", "EPITOPE", "DATASET"]).copy()
    epitope_to_label = {epitope: idx for idx, epitope in enumerate(MAIN_EPITOPES)}
    df.loc[:, "EPITOPE_LABELS"] = df["EPITOPE"].map(epitope_to_label).astype(int)
    return df.reset_index(drop=True)


def oversample_epitope(
    df: pd.DataFrame,
    epitope_col: str = "EPITOPE",
    group_col: str = "CLONOTYPE",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Mirror Holt et al. oversampling so each training epitope has equal count."""

    max_count = df[epitope_col].value_counts().max()
    rng = random_state

    def _oversample_single_epitope(epitope_df: pd.DataFrame) -> pd.DataFrame:
        needed = max_count - len(epitope_df)
        if needed <= 0:
            return epitope_df

        if not group_col or group_col not in epitope_df.columns:
            sampled = epitope_df.sample(n=needed, replace=True, random_state=rng)
            return pd.concat([epitope_df, sampled], ignore_index=True)

        pieces = [epitope_df]
        groups = list(epitope_df.groupby(group_col))
        base = needed // len(groups)
        remainder = needed % len(groups)
        for idx, (_, group_df) in enumerate(groups):
            group_needed = base + (1 if idx < remainder else 0)
            if group_needed:
                pieces.append(group_df.sample(n=group_needed, replace=True, random_state=rng))
        return pd.concat(pieces, ignore_index=True)

    return (
        df.groupby(epitope_col, group_keys=False)
        .apply(_oversample_single_epitope)
        .reset_index(drop=True)
    )


def split_dataframe(df: pd.DataFrame, oversample_train: bool = True, seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Return TRAIN/VAL/TEST frames plus an unbalanced TRAIN frame for evaluation."""

    splits = {split.lower(): df[df["DATASET"] == split].reset_index(drop=True) for split in DATASET_SPLITS}
    splits["train_eval"] = splits["train"]
    if oversample_train:
        train_df = oversample_epitope(splits["train"], random_state=seed)
        splits["train"] = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return splits


def get_antiberty_tokenizer(model_dir: str | Path | None = None) -> BertTokenizer:
    """Load AntiBERTy's local tokenizer from the installed `antiberty` package.

    The package layout is:
        trained_models/vocab.txt                  ← tokenizer vocab (one level up)
        trained_models/AntiBERTy_md_smooth/       ← model checkpoint directory
    """
    if model_dir is None:
        from importlib.resources import files

        # vocab.txt lives directly under trained_models/, not inside the checkpoint dir
        model_dir = files("antiberty").joinpath("trained_models")
    vocab_path = Path(model_dir) / "vocab.txt"
    if not vocab_path.exists():
        # Fallback: check one directory up in case a custom model_dir pointing to the
        # checkpoint directory was passed
        vocab_path = Path(model_dir).parent / "vocab.txt"
    return BertTokenizer(vocab_file=str(vocab_path), do_lower_case=False)


def get_esm2_tokenizer(model_id: str = ESM2_MODEL_ID):
    return AutoTokenizer.from_pretrained(model_id)


def _spaced(seq: str) -> str:
    return " ".join(list(seq))


def _build_esm2_input_ids(tokenizer, heavy: str, light: str, max_length: int) -> torch.Tensor:
    heavy_ids = tokenizer.convert_tokens_to_ids(list(heavy))
    light_ids = tokenizer.convert_tokens_to_ids(list(light))
    ids = [tokenizer.cls_token_id] + heavy_ids + [tokenizer.cls_token_id, tokenizer.cls_token_id] + light_ids + [
        tokenizer.eos_token_id
    ]
    return torch.tensor(ids[:max_length], dtype=torch.long)


def make_collate_fn(encoder: str, tokenizer, max_length: int = 512) -> Callable[[list[RBDExample]], dict[str, torch.Tensor]]:
    """Create an encoder-specific collator that tokenizes at batch time."""

    encoder = encoder.lower()
    if encoder not in {"antiberty", "esm2"}:
        raise ValueError(f"Unsupported encoder '{encoder}'. Expected 'antiberty' or 'esm2'.")

    def _collate_antiberty(batch: list[RBDExample]) -> dict[str, torch.Tensor]:
        tokens = tokenizer(
            [_spaced(example.heavy) for example in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {
            "h_input_ids": tokens["input_ids"].long(),
            "h_attention_mask": tokens["attention_mask"].long(),
            "labels": torch.tensor([example.label for example in batch], dtype=torch.long),
        }

    def _collate_esm2(batch: list[RBDExample]) -> dict[str, torch.Tensor]:
        input_ids = [_build_esm2_input_ids(tokenizer, example.heavy, example.light, max_length) for example in batch]
        padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = (padded != tokenizer.pad_token_id).long()
        return {
            "input_ids": padded.long(),
            "attention_mask": attention_mask,
            "labels": torch.tensor([example.label for example in batch], dtype=torch.long),
        }

    return _collate_antiberty if encoder == "antiberty" else _collate_esm2


def make_dataloaders(
    encoder: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    batch_size: int = 256,
    eval_batch_size: int | None = None,
    max_length: int = 512,
    num_workers: int = 0,
    oversample_train: bool = True,
    seed: int | None = None,
    tokenizer=None,
) -> dict[str, DataLoader]:
    """Build train, train_eval, val, and test dataloaders from `rbd_dataset.pd`."""

    eval_batch_size = eval_batch_size or batch_size
    encoder = encoder.lower()
    if tokenizer is None:
        tokenizer = get_antiberty_tokenizer() if encoder == "antiberty" else get_esm2_tokenizer()

    df = load_rbd_dataframe(data_path)
    split_frames = split_dataframe(df, oversample_train=oversample_train, seed=seed)
    collate_fn = make_collate_fn(encoder, tokenizer, max_length=max_length)

    return {
        "train": DataLoader(
            RBDDataset(split_frames["train"]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        "train_eval": DataLoader(
            RBDDataset(split_frames["train_eval"]),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            RBDDataset(split_frames["val"]),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            RBDDataset(split_frames["test"]),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
    }
