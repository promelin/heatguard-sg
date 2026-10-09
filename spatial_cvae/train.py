"""Train and validate the conditional spatial VAE."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .model import ConditionalSpatialVAE


BINARY_CHANNELS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
POSITIVE_WEIGHTS = [4.0, 3.0, 4.0, 8.0, 5.0, 3.0, 3.0, 4.0, 12.0, 12.0, 12.0, 8.0]


class SpatialLayoutDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.metadata = json.loads((root / "metadata.json").read_text())
        self.conditions = np.load(root / self.manifest["files"]["conditions"], mmap_mode="r")
        self.targets = np.load(root / self.manifest["files"]["targets"], mmap_mode="r")
        if len(self.conditions) != len(self.targets) or len(self.conditions) != len(self.metadata):
            raise ValueError("dataset arrays and metadata have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        condition = torch.from_numpy(np.array(self.conditions[index], copy=True)).float().div_(255.0)
        target = torch.from_numpy(np.array(self.targets[index], copy=True)).float().div_(255.0)
        return condition, target


def split_by_planning_area(metadata: list[dict], seed: int) -> dict[str, list[int]]:
    groups = sorted({row["planning_area"] for row in metadata})
    rng = random.Random(seed)
    rng.shuffle(groups)
    if len(groups) < 3:
        indexes = list(range(len(metadata)))
        rng.shuffle(indexes)
        val_count = max(1, int(len(indexes) * 0.15))
        test_count = max(1, int(len(indexes) * 0.10))
        return {
            "train": indexes[val_count + test_count :],
            "val": indexes[:val_count],
            "test": indexes[val_count : val_count + test_count],
        }
    val_count = max(1, round(len(groups) * 0.15))
    test_count = max(1, round(len(groups) * 0.10))
    val_groups = set(groups[:val_count])
    test_groups = set(groups[val_count : val_count + test_count])
    split = {"train": [], "val": [], "test": []}
    for index, row in enumerate(metadata):
        group = row["planning_area"]
        destination = "val" if group in val_groups else "test" if group in test_groups else "train"
        split[destination].append(index)
    return split


def spatial_vae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    site = condition[:, 0:1]
    binary_logits = logits[:, BINARY_CHANNELS]
    binary_target = target[:, BINARY_CHANNELS]
    bce = nn.functional.binary_cross_entropy_with_logits(binary_logits, binary_target, reduction="none")
    positive_weights = torch.tensor(
        POSITIVE_WEIGHTS, device=logits.device, dtype=logits.dtype
    ).view(1, -1, 1, 1)
    bce = bce * (1.0 + binary_target * (positive_weights - 1.0))
    binary_loss = (bce * site).sum() / (site.sum() * len(BINARY_CHANNELS) + 1e-6)

    probability = torch.sigmoid(binary_logits) * site
    truth = binary_target * site
    numerator = 2.0 * (probability * truth).sum(dim=(0, 2, 3)) + 1.0
    denominator = probability.sum(dim=(0, 2, 3)) + truth.sum(dim=(0, 2, 3)) + 1.0
    dice_loss = (1.0 - numerator / denominator).mean()

    building_mask = target[:, 0:1] * site
    height_loss = (
        torch.abs(torch.sigmoid(logits[:, 1:2]) - target[:, 1:2]) * building_mask
    ).sum() / (building_mask.sum() + 1e-6)
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
    building_probability = torch.sigmoid(logits[:, 0:1])
    hard_exclusion = condition[:, 5:6]
    constraint_loss = (building_probability * hard_exclusion * site).sum() / (
        (hard_exclusion * site).sum() + 1e-6
    )
    total = binary_loss + 0.55 * dice_loss + 1.5 * height_loss + beta * kl + 0.5 * constraint_loss
    return total, {
        "total": float(total.detach()),
        "binary_bce": float(binary_loss.detach()),
        "dice_loss": float(dice_loss.detach()),
        "height_l1": float(height_loss.detach()),
        "kl": float(kl.detach()),
        "constraint": float(constraint_loss.detach()),
        "beta": beta,
    }


@torch.no_grad()
def evaluate(
    model: ConditionalSpatialVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    channel_intersection = torch.zeros(len(BINARY_CHANNELS), device=device)
    channel_sum = torch.zeros(len(BINARY_CHANNELS), device=device)
    for condition, target in loader:
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mu, logvar = model.encode(condition, target)
        logits = model.decode(condition, mu)
        _, parts = spatial_vae_loss(logits, target, condition, mu, logvar, beta)
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + value
        site = condition[:, 0:1]
        prediction = (torch.sigmoid(logits[:, BINARY_CHANNELS]) >= 0.5).float() * site
        truth = target[:, BINARY_CHANNELS] * site
        channel_intersection += (prediction * truth).sum(dim=(0, 2, 3)) * 2
        channel_sum += prediction.sum(dim=(0, 2, 3)) + truth.sum(dim=(0, 2, 3))
        batches += 1
    if not batches:
        raise ValueError("validation loader is empty")
    result = {name: value / batches for name, value in totals.items()}
    dice = (channel_intersection + 1.0) / (channel_sum + 1.0)
    result["mean_binary_dice"] = float(dice.mean())
    for position, channel in enumerate(BINARY_CHANNELS):
        result[f"dice_channel_{channel}"] = float(dice[position])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--latent-injection", action="store_true")
    parser.add_argument("--skip-dropout", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.003)
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument("--diversity-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS training job did not receive a CUDA GPU")
    dataset = SpatialLayoutDataset(args.data.resolve())
    split = split_by_planning_area(dataset.metadata, args.seed)
    if not split["train"] or not split["val"]:
        raise ValueError(f"invalid spatial split: { {key: len(value) for key, value in split.items()} }")
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            Subset(dataset, indexes),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            generator=generator,
        )
        for name, indexes in split.items()
        if indexes
    }
    model_config = {
        "condition_channels": len(dataset.manifest["condition_names"]),
        "output_channels": len(dataset.manifest["target_names"]),
        "base_channels": args.base_channels,
        "latent_dim": args.latent_dim,
        "tile_size": dataset.manifest["tile_size"],
        "latent_injection": args.latent_injection,
        "condition_skip_dropout": args.skip_dropout,
    }
    model = ConditionalSpatialVAE(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch = 0
    best_loss = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_val_loss", best_loss)

    run_dir = args.output.resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    history_path = run_dir / "metrics.jsonl"
    no_improvement = 0
    global_step = 0
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_totals: dict[str, float] = {}
        batches = 0
        for condition, target in loaders["train"]:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            progress = min(1.0, (epoch + batches / max(1, len(loaders["train"]))) / max(1, args.kl_warmup_epochs))
            current_beta = args.beta * progress
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, mu, logvar = model(condition, target)
                loss, parts = spatial_vae_loss(
                    logits, target, condition, mu, logvar, current_beta
                )
                if args.diversity_weight > 0:
                    z_a = torch.randn(condition.shape[0], model.latent_dim, device=device)
                    z_b = torch.randn(condition.shape[0], model.latent_dim, device=device)
                    sample_a = torch.sigmoid(model.decode(condition, z_a))
                    sample_b = torch.sigmoid(model.decode(condition, z_b))
                    editable = condition[:, 1:2]
                    output_distance = (
                        torch.abs(sample_a - sample_b) * editable
                    ).sum() / (editable.sum() * model.output_channels + 1e-6)
                    diversity_loss = torch.relu(
                        torch.as_tensor(args.diversity_margin, device=device) - output_distance
                    )
                    loss = loss + args.diversity_weight * diversity_loss
                    parts["diversity_loss"] = float(diversity_loss.detach())
                    parts["sample_distance"] = float(output_distance.detach())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            for name, value in parts.items():
                epoch_totals[name] = epoch_totals.get(name, 0.0) + value
            batches += 1
            global_step += 1
        scheduler.step()
        train_metrics = {name: value / batches for name, value in epoch_totals.items()}
        val_metrics = evaluate(model, loaders["val"], device, args.beta)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": round(time.time() - started, 3),
            "train": train_metrics,
            "val": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        state = {
            "epoch": epoch,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": min(best_loss, val_metrics["total"]),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "dataset_manifest": dataset.manifest,
            "split_sizes": {name: len(indexes) for name, indexes in split.items()},
        }
        torch.save(state, run_dir / "last.pt")
        if val_metrics["total"] < best_loss:
            best_loss = val_metrics["total"]
            no_improvement = 0
            torch.save(state, run_dir / "best.pt")
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                print(f"EARLY_STOP epoch={epoch} best_val_loss={best_loss:.6f}")
                break
    summary = {
        "run_name": args.run_name,
        "best_val_loss": best_loss,
        "epochs_completed": epoch + 1,
        "device": str(device),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": round(time.time() - started, 3),
        "split_sizes": {name: len(indexes) for name, indexes in split.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
