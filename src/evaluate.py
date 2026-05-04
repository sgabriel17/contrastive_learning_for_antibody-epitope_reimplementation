"""Evaluation utilities for RBD contrastive checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import auc, average_precision_score, balanced_accuracy_score, f1_score
from tqdm.auto import tqdm

try:
    from .data_loader import DEFAULT_DATA_PATH, MAIN_EPITOPES, make_dataloaders
except ImportError:
    from data_loader import DEFAULT_DATA_PATH, MAIN_EPITOPES, make_dataloaders


def _as_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor


# ---------------------------------------------------------------------------
# Embedding collection
# ---------------------------------------------------------------------------

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

    # torch.no_grad() instead of inference_mode: ESM-2's RotaryEmbedding lazily
    # caches _cos/_sin tensors; if the cache is rebuilt inside inference_mode()
    # (e.g. due to a longer eval sequence) those tensors become unusable in the
    # next training forward/backward pass, crashing autograd.  no_grad avoids
    # gradient tracking with the same speed benefit while keeping tensors normal.
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding", leave=False):
            batch_labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp_enabled):
                batch_embeddings = model(**inputs)
            embeddings.append(batch_embeddings.float().cpu())
            labels.append(batch_labels.cpu())

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0).numpy()


# ---------------------------------------------------------------------------
# Per-epitope weighted AUROC  (Holt et al. method)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Threshold selection on validation set (paper protocol)
# ---------------------------------------------------------------------------

def find_threshold_on_val(
    train_embeddings: torch.Tensor | np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: torch.Tensor | np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """Find the cosine-similarity threshold that maximises balanced accuracy on
    train-vs-val pairs.  This threshold is then fixed and applied to the test set,
    matching the protocol in Holt et al. (implementation_plan.md §Evaluation Metrics).
    """
    emb1 = _as_numpy(train_embeddings)
    emb2 = _as_numpy(val_embeddings)
    scores = emb1 @ emb2.T
    y_true = (np.asarray(train_labels)[:, np.newaxis] == np.asarray(val_labels)).ravel().astype(int)
    y_score = scores.ravel()

    best_threshold = 0.0
    best_balanced_acc = -1.0
    for threshold in np.linspace(-1, 1, 1001):
        y_pred = y_score > threshold
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = float(balanced_acc)
            best_threshold = float(threshold)

    return best_threshold


# ---------------------------------------------------------------------------
# Pair-level scoring helpers
# ---------------------------------------------------------------------------

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


def _threshold_metrics_at(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute F1 and balanced accuracy at a *fixed* threshold (chosen on val)."""
    y_pred = y_score > threshold
    return {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Per-mode evaluation
# ---------------------------------------------------------------------------

def evaluate_pair_mode(
    embeddings1: torch.Tensor,
    labels1: np.ndarray,
    embeddings2: torch.Tensor,
    labels2: np.ndarray,
    same_dataset: bool,
    threshold: float,
) -> dict[str, float]:
    y_true, y_score = _binary_pair_scores(
        embeddings1, labels1, embeddings2, labels2, same_dataset=same_dataset
    )
    metrics = _threshold_metrics_at(y_true, y_score, threshold)
    metrics["weighted_auroc"] = get_cross_dataset_weighted_rocauc(
        embeddings1, labels1, embeddings2, labels2, same_dataset=same_dataset
    )
    return metrics


def evaluate_embedding_sets(
    train_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    val_embeddings: torch.Tensor,
    val_labels: np.ndarray,
    test_embeddings: torch.Tensor,
    test_labels: np.ndarray,
) -> dict[str, Any]:
    """Full evaluation following Holt et al. protocol:

    1. Threshold is chosen to maximise balanced accuracy on train-vs-val.
    2. That threshold is fixed and applied to train-vs-test and test-vs-test.
    3. AUROC / avg-precision are threshold-independent and always exact.
    """
    threshold = find_threshold_on_val(train_embeddings, train_labels, val_embeddings, val_labels)
    return {
        "val_threshold": float(threshold),
        "train_vs_test": evaluate_pair_mode(
            train_embeddings, train_labels, test_embeddings, test_labels,
            same_dataset=False, threshold=threshold,
        ),
        "test_vs_test": evaluate_pair_mode(
            test_embeddings, test_labels, test_embeddings, test_labels,
            same_dataset=True, threshold=threshold,
        ),
    }


# ---------------------------------------------------------------------------
# t-SNE visualisation + k-means accuracy
# ---------------------------------------------------------------------------

def plot_tsne(
    embeddings: torch.Tensor | np.ndarray,
    labels: np.ndarray,
    title: str = "RBD antibody embeddings",
    perplexity: int = 30,
    save_path: str | Path | None = None,
    label_names: list[str] | None = None,
) -> dict[str, float]:
    """Run t-SNE, plot coloured by epitope, compute k-means accuracy, and optionally save.

    Returns a dict with ``kmeans_accuracy`` (k=12, matching Holt et al.).
    """
    emb = _as_numpy(embeddings)
    if label_names is None:
        label_names = MAIN_EPITOPES

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(emb)

    # k-means accuracy (unsupervised, k=12)
    km = KMeans(n_clusters=len(label_names), n_init=10, random_state=42)
    km_labels = km.fit_predict(emb)
    # best-matching permutation via majority vote per cluster
    km_acc = _kmeans_accuracy(labels, km_labels, n_clusters=len(label_names))

    # Plot
    label_str = np.array(label_names)[np.asarray(labels)]
    palette = {name: mcolors.rgb2hex(color)
               for name, color in zip(label_names, sns.color_palette("husl", len(label_names)))}

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=label_str,
                    palette=palette, s=18, linewidth=0, ax=ax, legend="full")
    ax.set_title(f"{title}  |  k-means acc = {km_acc:.3f}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Epitope", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {"kmeans_accuracy": float(km_acc)}


def plot_cosine_distributions(
    panels: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Cosine Similarity Distributions",
    save_path: str | Path | None = None,
) -> None:
    """Plot same- vs different-epitope cosine-similarity histograms.

    Matches the Holt et al. Figure 3A style: shared x-axis [−1, 1], decision
    threshold line, and balanced accuracy annotation per panel.

    Parameters
    ----------
    panels : dict mapping panel title → (y_true, y_score) arrays where
        y_true is 0/1 (same epitope) and y_score is the cosine similarity.
    title : overall figure title.
    save_path : if given, save PNG at this path (parent dirs created automatically).
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (panel_name, (y_true, y_score)) in zip(axes, panels.items()):
        same_ep = y_score[y_true == 1]
        diff_ep = y_score[y_true == 0]

        ax.hist(diff_ep, bins=100, alpha=0.6, density=True,
                label="Different Epitopes", color="#EF5350")
        ax.hist(same_ep, bins=100, alpha=0.6, density=True,
                label="Same Epitope", color="#42A5F5")

        best_thresh, best_bacc = 0.0, -1.0
        for t in np.linspace(-1, 1, 1001):
            bacc = balanced_accuracy_score(y_true, y_score > t)
            if bacc > best_bacc:
                best_bacc, best_thresh = bacc, t

        ax.axvline(best_thresh, color="black", linestyle="--", linewidth=1.5,
                   label="Decision Threshold")
        ax.set_title(f"{panel_name}\n{best_bacc * 100:.1f}% Accuracy", fontsize=11)
        ax.set_xlabel("Cosine Similarity", fontsize=11)
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Density", fontsize=11)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
    plt.close(fig)


def _kmeans_accuracy(true_labels: np.ndarray, km_labels: np.ndarray, n_clusters: int) -> float:
    """Map each k-means cluster to its majority true label and compute accuracy."""
    mapping = {}
    for cluster in range(n_clusters):
        mask = km_labels == cluster
        if not np.any(mask):
            mapping[cluster] = -1
            continue
        counts = np.bincount(true_labels[mask], minlength=n_clusters)
        mapping[cluster] = int(np.argmax(counts))
    predicted = np.array([mapping[c] for c in km_labels])
    return float((predicted == true_labels).mean())


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint_model(
    checkpoint_path: str | Path,
    encoder: str,
    device: torch.device,
    load_in_4bit: bool | None = None,
):
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


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str | Path,
    encoder: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    batch_size: int = 256,
    max_length: int = 512,
    load_in_4bit: bool | None = None,
    output_dir: str | Path | None = None,
    plot: bool = False,
    perplexity: int = 30,
) -> dict[str, Any]:
    """Load a checkpoint, embed train/val/test, apply the val-threshold protocol,
    compute all metrics, and optionally save a t-SNE figure.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = make_dataloaders(
        encoder=encoder,
        data_path=data_path,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        max_length=max_length,
        oversample_train=False,
    )
    model = load_checkpoint_model(
        checkpoint_path, encoder=encoder, device=device, load_in_4bit=load_in_4bit
    )

    print("Embedding train split …")
    train_embeddings, train_labels = collect_embeddings(model, dataloaders["train_eval"], device, use_amp=True)
    print("Embedding val split …")
    val_embeddings, val_labels = collect_embeddings(model, dataloaders["val"], device, use_amp=True)
    print("Embedding test split …")
    test_embeddings, test_labels = collect_embeddings(model, dataloaders["test"], device, use_amp=True)

    results: dict[str, Any] = evaluate_embedding_sets(
        train_embeddings, train_labels,
        val_embeddings, val_labels,
        test_embeddings, test_labels,
    )

    if plot:
        out = Path(output_dir) if output_dir else Path(checkpoint_path).parent
        out.mkdir(parents=True, exist_ok=True)

        # Test-set t-SNE (most informative)
        tsne_metrics = plot_tsne(
            test_embeddings, test_labels,
            title=f"{encoder} — test set",
            perplexity=perplexity,
            save_path=out / "tsne_test.png",
        )
        results["tsne_test"] = tsne_metrics

        # Train+test combined (matches Holt et al. figure style)
        combined_emb = torch.cat([train_embeddings, test_embeddings], dim=0)
        combined_labels = np.concatenate([train_labels, test_labels])
        tsne_metrics_combined = plot_tsne(
            combined_emb, combined_labels,
            title=f"{encoder} — train + test",
            perplexity=perplexity,
            save_path=out / "tsne_train_test.png",
        )
        results["tsne_train_test"] = tsne_metrics_combined

        # Cosine similarity distributions (Holt et al. Figure 3A style)
        y_true_tvt, y_score_tvt = _binary_pair_scores(
            train_embeddings, train_labels, test_embeddings, test_labels, same_dataset=False,
        )
        y_true_tt, y_score_tt = _binary_pair_scores(
            test_embeddings, test_labels, test_embeddings, test_labels, same_dataset=True,
        )
        plot_cosine_distributions(
            {
                f"{encoder} (finetuned) Train vs Test": (y_true_tvt, y_score_tvt),
                f"{encoder} (finetuned) Test vs Test": (y_true_tt, y_score_tt),
            },
            title=f"Cosine Similarity — {encoder} (finetuned)",
            save_path=out / "cosine_distributions.png",
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a contrastive RBD checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", choices=["antiberty", "esm2"], required=True)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_dir", default="",
                        help="Directory to save figures and JSON (defaults to checkpoint's parent dir).")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save t-SNE figures.")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or str(Path(args.checkpoint).parent)

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        encoder=args.encoder,
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
        output_dir=output_dir,
        plot=args.plot,
        perplexity=args.perplexity,
    )

    print(json.dumps(metrics, indent=2))

    out_json = args.output_json or str(Path(output_dir) / "final_eval.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
