from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Small vocabulary helper
# ---------------------------------------------------------------------------


@dataclass
class Vocab:
    """Minimal whitespace vocabulary with JSON serialisation.

    The goal here is not to be clever but to have something transparent that
    you can easily explain in the thesis. Tokenisation is intentionally kept
    as lowercase + whitespace split.
    """

    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int = 0
    unk_id: int = 1

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return text.lower().strip().split()

    @classmethod
    def build(cls, texts: Sequence[str], min_freq: int = 2) -> "Vocab":
        freq: Dict[str, int] = {}
        for t in texts:
            for tok in cls._tokenise(t):
                if not tok:
                    continue
                freq[tok] = freq.get(tok, 0) + 1

        # Reserve 0 for PAD and 1 for UNK.
        stoi: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        for tok, c in sorted(freq.items()):
            if c < min_freq:
                continue
            if tok not in stoi:
                stoi[tok] = len(stoi)

        itos = [""] * len(stoi)
        for tok, idx in stoi.items():
            itos[idx] = tok

        return cls(stoi=stoi, itos=itos, pad_id=0, unk_id=1)

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = self._tokenise(text)
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokens[: max_len]]
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids

    def to_json(self) -> Dict[str, object]:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
        }

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "Vocab":
        return cls(
            stoi={str(k): int(v) for k, v in (data.get("stoi") or {}).items()},
            itos=[str(x) for x in (data.get("itos") or [])],
            pad_id=int(data.get("pad_id", 0)),
            unk_id=int(data.get("unk_id", 1)),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_json(data)

    def __len__(self) -> int:  # pragma: no cover - tiny helper
        return len(self.stoi)


# ---------------------------------------------------------------------------
# Transformer-based text classifier
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer encoders."""

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings to the input.

        Args:
            x: (batch, seq_len, dim)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class TextTransformerClassifier(nn.Module):
    """Tiny Transformer encoder followed by mean pooling and a linear head.

    This is intentionally compact and readable; the idea is that you can
    expand or tweak the hyper-parameters during experimentation without
    having to change the API integration.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        emb_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, token_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Encode a batch and return per-class logits.

        Args:
            token_ids: (batch, seq_len) integer ids.
            padding_mask: (batch, seq_len) bool mask, True for padding.

        Returns:
            (batch, num_classes) unnormalised scores.
        """
        x = self.emb(token_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)

        if padding_mask is not None:
            # Mask out padding positions when averaging.
            mask = (~padding_mask).float()  # 1.0 for real tokens
            summed = (x * mask.unsqueeze(-1)).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / lengths.unsqueeze(-1)
        else:
            pooled = x.mean(dim=1)

        logits = self.classifier(pooled)
        return logits


# ---------------------------------------------------------------------------
# Checkpoint helpers for training / inference
# ---------------------------------------------------------------------------


def save_checkpoint(
    output_dir: Path,
    model: TextTransformerClassifier,
    vocab: Vocab,
    label_to_index: Dict[str, int],
    config: Dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model weights and config
    ckpt = {
        "model_state": model.state_dict(),
        "config": config,
    }
    torch.save(ckpt, output_dir / "text_transformer.pt")

    # Vocab and labels
    vocab.save(output_dir / "vocab.json")
    (output_dir / "labels.json").write_text(
        json.dumps(label_to_index, indent=2, sort_keys=True)
    )


def load_inference_artifacts(
    model_dir: Path,
    device: torch.device | str = "cpu",
) -> Tuple[TextTransformerClassifier, Vocab, List[str], Dict[str, object]]:
    """Load model, vocab and label mapping for inference.

    Returns:
        (model, vocab, idx_to_label, config)
    """
    model_dir = Path(model_dir)

    ckpt_path = model_dir / "text_transformer.pt"
    vocab_path = model_dir / "vocab.json"
    labels_path = model_dir / "labels.json"

    if not (ckpt_path.exists() and vocab_path.exists() and labels_path.exists()):
        raise FileNotFoundError(
            f"Expected model artifacts not found in {model_dir}. "
            "Run the training script before using the API."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    num_classes = int(config.get("num_classes", 1))
    vocab = Vocab.load(vocab_path)
    vocab_size = len(vocab)

    model = TextTransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        emb_dim=int(config.get("emb_dim", 128)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 256)),
        dropout=float(config.get("dropout", 0.1)),
        max_len=int(config.get("max_len", 128)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    label_to_index = json.loads(labels_path.read_text())
    # We want index -> label for ranking.
    idx_to_label: List[str] = [""] * len(label_to_index)
    for label, idx in label_to_index.items():
        idx_to_label[int(idx)] = label

    return model, vocab, idx_to_label, config
