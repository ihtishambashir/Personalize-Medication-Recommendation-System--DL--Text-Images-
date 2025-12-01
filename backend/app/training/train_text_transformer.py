from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

from app.models.text_transformer import (
    TextTransformerClassifier,
    Vocab,
    save_checkpoint,
)


class TextDataset(Dataset):
    """Simple (text, label) dataset backed by tensors."""

    def __init__(self, inputs: Tensor, labels: Tensor) -> None:
        assert inputs.size(0) == labels.size(0)
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.inputs[idx], self.labels[idx]


def load_medicine_csv(path: Path) -> Tuple[List[str], List[str]]:
    """Read the medicine CSV and return (texts, reasons)."""
    df = pd.read_csv(path)
    # Be defensive about column names and missing values.
    for col in ("Reason", "Description", "Drug_Name"):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {path}, found {list(df.columns)}")

    # Drop rows without a reason.
    df = df.dropna(subset=["Reason"])
    df["Reason"] = df["Reason"].astype(str).str.strip()

    # Choose description if available, otherwise fall back to the drug name.
    desc = df["Description"].fillna("").astype(str).str.strip()
    names = df["Drug_Name"].fillna("").astype(str).str.strip()
    texts = [
        (d if d else n) or ""
        for d, n in zip(desc.tolist(), names.tolist())
    ]
    labels = df["Reason"].tolist()
    return texts, labels


def build_label_mapping(labels: List[str]) -> Tuple[dict, list]:
    unique = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique)}
    idx_to_label = unique
    return label_to_index, idx_to_label


def encode_texts(
    texts: List[str],
    vocab: Vocab,
    max_len: int,
) -> Tensor:
    encoded = [vocab.encode(t, max_len=max_len) for t in texts]
    return torch.tensor(encoded, dtype=torch.long)


def train(
    data_path: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 64,
    max_len: int = 64,
    lr: float = 1e-3,
    min_freq: int = 2,
    val_ratio: float = 0.2,
    device: str = "cpu",
) -> None:
    texts, labels = load_medicine_csv(data_path)
    if not texts:
        raise RuntimeError("No training examples found in the CSV file.")

    label_to_index, idx_to_label = build_label_mapping(labels)
    y = torch.tensor([label_to_index[lbl] for lbl in labels], dtype=torch.long)

    # Build vocabulary on the full corpus (for this demo that is acceptable).
    vocab = Vocab.build(texts, min_freq=min_freq)

    X = encode_texts(texts, vocab, max_len=max_len)

    dataset = TextDataset(X, y)

    # Train/validation split.
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = max(1, n_total - n_val)
    if n_train <= 0:
        raise RuntimeError("Not enough data for a train/validation split.")

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device_t = torch.device(device)
    model = TextTransformerClassifier(
        vocab_size=len(vocab),
        num_classes=len(idx_to_label),
        emb_dim=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=max_len,
    ).to(device_t)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    train_acc_history: List[float] = []
    val_acc_history: List[float] = []

    for epoch in range(1, epochs + 1):
        # -----------------------------
        # Training
        # -----------------------------
        model.train()
        correct = 0
        total = 0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device_t)
            batch_labels = batch_labels.to(device_t)
            padding_mask = batch_inputs.eq(0)  # 0 is PAD

            optimiser.zero_grad()
            logits = model(batch_inputs, padding_mask=padding_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimiser.step()

            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        train_acc = correct / max(1, total)
        train_acc_history.append(train_acc)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                batch_inputs = batch_inputs.to(device_t)
                batch_labels = batch_labels.to(device_t)
                padding_mask = batch_inputs.eq(0)

                logits = model(batch_inputs, padding_mask=padding_mask)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

        val_acc = correct / max(1, total)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} - train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    # ------------------------------------------------------------------
    # Save artefacts for the API + thesis
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model + vocab + labels.
    config = {
        "vocab_size": len(vocab),
        "num_classes": len(idx_to_label),
        "emb_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "max_len": max_len,
    }
    save_checkpoint(output_dir, model, vocab, label_to_index, config)

    # Accuracy curves as raw data
    np.save(output_dir / "train_acc.npy", np.array(train_acc_history, dtype=np.float32))
    np.save(output_dir / "val_acc.npy", np.array(val_acc_history, dtype=np.float32))

    # And as a PNG figure for the thesis.
    try:
        import matplotlib.pyplot as plt

        epochs_axis = range(1, epochs + 1)
        plt.figure()
        plt.plot(epochs_axis, train_acc_history, label="train")
        plt.plot(epochs_axis, val_acc_history, label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Text Transformer accuracy on medicine.csv")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_curve.png")
        plt.close()
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Could not generate accuracy plot: {exc}")

    print(f"Training complete. Artefacts saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small Transformer on medicine.csv.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("dataset/text/medicine.csv"),
        help="Path to medicine.csv (text dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trained_models/text_transformer"),
        help="Directory where model and plots will be stored.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--max-len", type=int, default=64, help="Maximum number of tokens per example.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum token frequency to keep in the vocab.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of the data to use for validation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device (e.g. 'cpu' or 'cuda').",
    )

    args = parser.parse_args()
    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=args.lr,
        min_freq=args.min_freq,
        val_ratio=args.val_ratio,
        device=args.device,
    )


if __name__ == "__main__":
    main()
