"""Train AntiBERTy or ESM-2 with supervised contrastive QLoRA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

try:
    from .data_loader import DEFAULT_DATA_PATH, make_dataloaders
    from .evaluate import collect_embeddings, evaluate_pair_mode
    from .loss import ContrastiveLoss
    from .models import build_model, move_model_to_device
except ImportError:
    from data_loader import DEFAULT_DATA_PATH, make_dataloaders
    from evaluate import collect_embeddings, evaluate_pair_mode
    from loss import ContrastiveLoss
    from models import build_model, move_model_to_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=enabled and device.type == "cuda")


def _model_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return move_model_to_device(model, device)


def _trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def _batch_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    labels = batch["labels"].to(device)
    inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
    return inputs, labels


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    metrics: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "rng_state": _rng_state(),
        },
        path,
    )


def load_training_checkpoint(
    resume_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[int, list[dict[str, Any]], float]:
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    _restore_rng_state(checkpoint.get("rng_state"))
    metrics = checkpoint.get("metrics", [])
    best_val = max(
        (entry.get("train_vs_val", {}).get("weighted_auroc", float("-inf")) for entry in metrics),
        default=float("-inf"),
    )
    return int(checkpoint["epoch"]) + 1, metrics, best_val


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion: ContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    use_amp: bool,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []

    for step, batch in enumerate(tqdm(dataloader, desc="Train", leave=False), start=1):
        inputs, labels = _batch_inputs(batch, device)
        with _autocast(device, use_amp):
            embeddings = model(**inputs)
            loss = criterion(embeddings, labels) / grad_accum_steps

        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()) * grad_accum_steps)

    if len(dataloader) % grad_accum_steps:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return float(np.mean(losses))


def evaluate_validation(model: torch.nn.Module, dataloaders, device: torch.device, use_amp: bool) -> dict[str, dict[str, float]]:
    train_embeddings, train_labels = collect_embeddings(model, dataloaders["train_eval"], device, use_amp=use_amp)
    val_embeddings, val_labels = collect_embeddings(model, dataloaders["val"], device, use_amp=use_amp)
    return {
        "train_vs_val": evaluate_pair_mode(
            train_embeddings,
            train_labels,
            val_embeddings,
            val_labels,
            same_dataset=False,
        ),
        "val_vs_val": evaluate_pair_mode(
            val_embeddings,
            val_labels,
            val_embeddings,
            val_labels,
            same_dataset=True,
        ),
    }


def write_metrics(output_dir: Path, metrics: list[dict[str, Any]]) -> None:
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning for RBD epitope embeddings.")
    parser.add_argument("--encoder", choices=["antiberty", "esm2"], required=True)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output_dir", required=True, help="Use a persistent path on Colab, e.g. Google Drive.")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.3)
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--no_amp", action="store_true")
    parser.set_defaults(load_in_4bit=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = make_dataloaders(
        encoder=args.encoder,
        data_path=args.data_path,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        oversample_train=True,
        seed=args.seed,
    )
    model = build_model(
        args.encoder,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
    )
    model = _model_to_device(model, device)
    trainable, total = _trainable_parameter_count(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr)
    criterion = ContrastiveLoss(temperature=args.temperature)
    use_amp = torch.cuda.is_available() and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    metrics: list[dict[str, Any]] = []
    best_val_auroc = float("-inf")
    if args.resume:
        start_epoch, metrics, best_val_auroc = load_training_checkpoint(args.resume, model, optimizer, scaler)
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        train_loss = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=use_amp,
        )
        epoch_metrics: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            epoch_metrics.update(evaluate_validation(model, dataloaders, device, use_amp=use_amp))
            val_auroc = epoch_metrics["train_vs_val"]["weighted_auroc"]
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, scaler, epoch, metrics + [epoch_metrics], args)

        metrics.append(epoch_metrics)
        write_metrics(output_dir, metrics)
        save_checkpoint(output_dir / "checkpoint_latest.pt", model, optimizer, scaler, epoch, metrics, args)
        if args.checkpoint_every > 0 and (epoch % args.checkpoint_every == 0 or epoch == 1):
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, scaler, epoch, metrics, args)

        status = f"epoch={epoch} train_loss={train_loss:.4f}"
        if "train_vs_val" in epoch_metrics:
            status += f" train_vs_val_weighted_auroc={epoch_metrics['train_vs_val']['weighted_auroc']:.4f}"
        print(status)


if __name__ == "__main__":
    main()
