from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.models.vision_transformer import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_vit_model,
    save_vit_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """Create train/val/test dataloaders from the folder structure."""
    data_root = Path(data_root)
    splits = {"train": data_root / "train", "val": data_root / "val", "test": data_root / "test"}
    for split, path in splits.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected directory for split '{split}' at {path}")

    train_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_ds = datasets.ImageFolder(splits["train"], transform=train_tfms)
    val_ds = datasets.ImageFolder(splits["val"], transform=eval_tfms)
    test_ds = datasets.ImageFolder(splits["test"], transform=eval_tfms)

    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError("Class mappings differ across splits. Please ensure identical class folders.")

    class_to_idx = train_ds.class_to_idx

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders, class_to_idx


def train_and_eval(
    data_root: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    image_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    num_workers: int,
    pretrained: bool,
    device: str,
    seed: int,
) -> None:
    set_seed(seed)
    device_t = torch.device(device)

    loaders, class_to_idx = build_dataloaders(
        data_root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    idx_to_class: List[str] = [""] * len(class_to_idx)
    for cls_name, idx in class_to_idx.items():
        idx_to_class[idx] = cls_name

    model = create_vit_model(
        num_classes=len(class_to_idx),
        pretrained=pretrained,
        dropout=dropout,
    ).to(device_t)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    train_acc_history: List[float] = []
    val_acc_history: List[float] = []

    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for images, labels in loaders["train"]:
            images = images.to(device_t)
            labels = labels.to(device_t)

            optimiser.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in loaders["val"]:
                images = images.to(device_t)
                labels = labels.to(device_t)
                logits = model(images)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)

                val_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce any checkpoint.")

    # Load best weights before testing.
    model.load_state_dict(best_state["model_state"])
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    per_class_correct = [0 for _ in range(len(idx_to_class))]
    per_class_total = [0 for _ in range(len(idx_to_class))]

    with torch.no_grad():
        for images, labels in loaders["test"]:
            images = images.to(device_t)
            labels = labels.to(device_t)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            test_loss += loss.item() * labels.size(0)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

            for p, t in zip(preds.tolist(), labels.tolist()):
                per_class_total[t] += 1
                if p == t:
                    per_class_correct[t] += 1

    test_loss = test_loss / max(1, test_total)
    test_acc = test_correct / max(1, test_total)
    per_class_acc = {
        idx_to_class[i]: (per_class_correct[i] / per_class_total[i] if per_class_total[i] else 0.0)
        for i in range(len(idx_to_class))
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "train_loss": train_loss_history,
        "train_acc": train_acc_history,
        "val_loss": val_loss_history,
        "val_acc": val_acc_history,
        "best_val_acc": best_val_acc,
        "best_epoch": int(best_state["epoch"]),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "per_class_accuracy": per_class_acc,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save checkpoint + mappings.
    config = {
        "image_size": image_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "pretrained": pretrained,
        "num_classes": len(class_to_idx),
        "best_val_acc": best_val_acc,
        "seed": seed,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }
    save_vit_checkpoint(output_dir, model, class_to_idx, config)

    # Plots for thesis/reporting.
    epochs_axis = range(1, epochs + 1)
    plt.figure()
    plt.plot(epochs_axis, train_acc_history, label="train")
    plt.plot(epochs_axis, val_acc_history, label="validation")
    plt.axhline(test_acc, color="black", linestyle="--", label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ViT accuracy on skin images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs_axis, train_loss_history, label="train")
    plt.plot(epochs_axis, val_loss_history, label="validation")
    plt.axhline(test_loss, color="black", linestyle="--", label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ViT loss on skin images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    print(f"Training complete. Best val acc={best_val_acc:.4f}, test acc={test_acc:.4f}")
    print(f"Artifacts saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ViT classifier on the skin image dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/skin images"),
        help="Root directory containing train/val/test subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trained_models/vision_transformer"),
        help="Directory where checkpoints and plots will be written.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution for the ViT.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout before the classification head.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet-pretrained weights for faster convergence.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    train_and_eval(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
