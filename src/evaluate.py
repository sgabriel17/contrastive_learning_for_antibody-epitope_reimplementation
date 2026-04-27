"""Evaluation utilities for RBD contrastive checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, balanced_accuracy_score, f1_score
from tqdm.auto import tqdm

try:
    from .data_loader import DEFAULT_DATA_PATH, make_dataloaders
except ImportError:
    from data_loader import DEFAULT_DATA_PATH, make_dataloaders


def _as_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor


def collect_embeddings(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[torch.Tensor, np.ndarray]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    amp_enabled = use_amp and device.type == "cuda"

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Embedding", leave=False):
            batch_labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp_enabled):
                batch_embeddings = model(**inputs)
            embeddings.append(batch_embeddings.float().cpu())
            labels.append(batch_labels.cpu())

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0).numpy()


def get_cross_dataset_weighted_rocauc(
    embeddings1: torch.Tensor | np.ndarray,
    labels1: np.ndarray,
    embeddings2: torch.Tensor | np.ndarray,
    labels2: np.ndarray,
    same_dataset: bool = False,
) -> float:
    """Per-epitope weighted AUROC, following Holt et al.'s threshold-curve method."""

    emb1 = _as_numpy(embeddings1)
    emb2 = _as_numpy(embeddings2)
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    num_epitopes = int(max(labels1.max(), labels2.max()) + 1)
    thresholds = np.linspace(-1, 1, 1001)
    tprs: list[np.ndarray] = []
    fprs: list[np.ndarray] = []
    weights: list[float] = []

    for epitope in range(num_epitopes):
        ep_filter = labels1 == epitope
        if not np.any(ep_filter):
            continue

        anchor_embeddings = emb1[ep_filter]
        anchor_labels = labels1[ep_filter]
        comparisons = anchor_embeddings @ emb2.T
        actual = anchor_labels[:, np.newaxis] == labels2
        valid = np.ones_like(actual, dtype=bool)
        if same_dataset:
            anchor_indices = np.flatnonzero(ep_filter)
            valid &= anchor_indices[:, np.newaxis] != np.arange(len(labels2))[np.newaxis, :]

        ntrue = np.logical_and(actual, valid).sum()
        nfalse = np.logical_and(~actual, valid).sum()
        if ntrue == 0 or nfalse == 0:
            continue

        guesses = comparisons[:, :, np.newaxis] > thresholds
        valid_3d = valid[:, :, np.newaxis]
        tprs.append((guesses & actual[:, :, np.newaxis] & valid_3d).sum(axis=(0, 1)) / ntrue)
        fprs.append((guesses & ~actual[:, :, np.newaxis] & valid_3d).sum(axis=(0, 1)) / nfalse)
        weights.append(float(ntrue))

    if not weights:
        return float("nan")

    weighted_tpr = np.average(np.stack(tprs), axis=0, weights=np.asarray(weights))
    weighted_fpr = np.average(np.stack(fprs), axis=0, weights=np.asarray(weights))
    order = np.argsort(weighted_fpr)
    return float(auc(weighted_fpr[order], weighted_tpr[order]))


def _binary_pair_scores(
    embeddings1: torch.Tensor | np.ndarray,
    labels1: np.ndarray,
    embeddings2: torch.Tensor | np.ndarray,
    labels2: np.ndarray,
    same_dataset: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    emb1 = _as_numpy(embeddings1)
    emb2 = _as_numpy(embeddings2)
    scores = emb1 @ emb2.T
    y_true = np.asarray(labels1)[:, np.newaxis] == np.asarray(labels2)
    valid = np.ones_like(y_true, dtype=bool)
    if same_dataset:
        valid &= ~np.eye(len(labels1), len(labels2), dtype=bool)
    return y_true[valid].astype(int), scores[valid]


def _threshold_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    thresholds = np.linspace(-1, 1, 1001)
    best_threshold = 0.0
    best_balanced_acc = -1.0
    for threshold in thresholds:
        y_pred = y_score > threshold
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = float(balanced_acc)
            best_threshold = float(threshold)

    y_pred = y_score > best_threshold
    return {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": best_balanced_acc,
        "threshold": best_threshold,
    }


def evaluate_pair_mode(
    embeddings1: torch.Tensor,
    labels1: np.ndarray,
    embeddings2: torch.Tensor,
    labels2: np.ndarray,
    same_dataset: bool,
) -> dict[str, float]:
    y_true, y_score = _binary_pair_scores(embeddings1, labels1, embeddings2, labels2, same_dataset=same_dataset)
    metrics = _threshold_metrics(y_true, y_score)
    metrics["weighted_auroc"] = get_cross_dataset_weighted_rocauc(
        embeddings1,
        labels1,
        embeddings2,
        labels2,
        same_dataset=same_dataset,
    )
    return metrics


def evaluate_embedding_sets(
    train_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    test_embeddings: torch.Tensor,
    test_labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        "train_vs_test": evaluate_pair_mode(
            train_embeddings,
            train_labels,
            test_embeddings,
            test_labels,
            same_dataset=False,
        ),
        "test_vs_test": evaluate_pair_mode(
            test_embeddings,
            test_labels,
            test_embeddings,
            test_labels,
            same_dataset=True,
        ),
    }


def load_checkpoint_model(checkpoint_path: str | Path, encoder: str, device: torch.device, load_in_4bit: bool | None = None):
    try:
        from .models import build_model, move_model_to_device
    except ImportError:
        from models import build_model, move_model_to_device

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args: dict[str, Any] = checkpoint.get("args", {})
    model = build_model(
        encoder,
        lora_r=int(args.get("lora_r", 16)),
        lora_alpha=int(args.get("lora_alpha", 32)),
        lora_dropout=float(args.get("lora_dropout", 0.3)),
        load_in_4bit=load_in_4bit if load_in_4bit is not None else args.get("load_in_4bit"),
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return move_model_to_device(model, device)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    encoder: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    batch_size: int = 256,
    max_length: int = 512,
    load_in_4bit: bool | None = None,
) -> dict[str, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = make_dataloaders(
        encoder=encoder,
        data_path=data_path,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        max_length=max_length,
        oversample_train=False,
    )
    model = load_checkpoint_model(checkpoint_path, encoder=encoder, device=device, load_in_4bit=load_in_4bit)
    train_embeddings, train_labels = collect_embeddings(model, dataloaders["train_eval"], device, use_amp=True)
    test_embeddings, test_labels = collect_embeddings(model, dataloaders["test"], device, use_amp=True)
    return evaluate_embedding_sets(train_embeddings, train_labels, test_embeddings, test_labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a contrastive RBD checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", choices=["antiberty", "esm2"], required=True)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        encoder=args.encoder,
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
    )
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
