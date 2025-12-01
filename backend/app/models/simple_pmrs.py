from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding used with Transformer encoders."""

    def __init__(self, dim: int, max_len: int = 256) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerTextEncoder(nn.Module):
    """Light-weight Transformer encoder for textual EHR information.

    It takes integer token ids as input and returns a single dense representation
    for the whole sequence (CLS-style pooling).
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
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

    def forward(self, token_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Encode a batch of token sequences.

        Args:
            token_ids: (batch, seq_len) integer indices into the vocabulary.
            padding_mask: optional boolean mask with shape (batch, seq_len) where
                True marks padding positions that should be ignored.

        Returns:
            Tensor of shape (batch, emb_dim) – one vector per sequence.
        """
        x = self.emb(token_ids)  # (batch, seq, emb_dim)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)

        # Simple pooling: mean over non-padded positions.
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = (x * (~padding_mask).unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = x.mean(dim=1)
        return pooled


class CNNImageEncoder(nn.Module):
    """Compact CNN encoder for dermatology images.

    The architecture is intentionally simple: a few convolution + pooling blocks
    followed by a projection layer. This keeps the dependency surface small
    while still giving you a proper learnable image encoder.
    """

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch of images.

        Args:
            images: Tensor of shape (batch, 3, H, W) with pixel values in [0, 1].

        Returns:
            Tensor of shape (batch, output_dim).
        """
        feats = self.features(images)  # (batch, 128, 1, 1)
        feats = feats.flatten(1)
        return self.proj(feats)


class SimplePMRSModel(nn.Module):
    """Multimodal PMRS backbone: Transformer for text + CNN for images.

    The goal is not to be state-of-the-art but to have a clear, readable example
    that matches the high-level architecture described in your thesis:

      * textual EHR (diagnoses, free-text notes) → TransformerTextEncoder
      * dermatology images → CNNImageEncoder
      * fused representation → medication scoring head

    The actual training will be done offline; the API only needs the forward pass.
    """

    def __init__(
        self,
        vocab_size: int,
        num_meds: int,
        text_dim: int = 128,
        image_dim: int = 128,
        hidden_dim: int = 64,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.text_encoder = TransformerTextEncoder(
            vocab_size=vocab_size,
            emb_dim=text_dim,
            num_heads=4,
            num_layers=2,
            dim_feedforward=4 * text_dim,
            dropout=0.1,
            max_len=max_len,
        )
        self.image_encoder = CNNImageEncoder(output_dim=image_dim)

        fusion_dim = text_dim + image_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_meds),
        )

    def forward(
        self,
        token_ids: Tensor,
        padding_mask: Tensor | None = None,
        images: Tensor | None = None,
    ) -> Tensor:
        """Compute raw medication scores (logits).

        Args:
            token_ids: (batch, seq_len) token ids.
            padding_mask: (batch, seq_len) bool mask, True for padding tokens.
            images: optional (batch, 3, H, W) image batch. If ``None`` a zero
                vector is used instead, effectively falling back to a text-only
                model.

        Returns:
            Tensor of shape (batch, num_meds) with unnormalised scores.
        """
        text_repr = self.text_encoder(token_ids, padding_mask=padding_mask)

        if images is not None:
            img_repr = self.image_encoder(images)
        else:
            img_repr = torch.zeros(
                text_repr.size(0),
                self.head[0].in_features - text_repr.size(1),
                device=text_repr.device,
                dtype=text_repr.dtype,
            )

        fused = torch.cat([text_repr, img_repr], dim=1)
        return self.head(fused)

    # ---------------------------------------------------------------------
    # Utilities for the demo setup
    # ---------------------------------------------------------------------
    @staticmethod
    def build_demo(max_vocab_size: int = 512) -> Tuple["SimplePMRSModel", Dict[str, int], List[str]]:
        """Construct a small demo instance, vocabulary and medication list.

        The function is deterministic and self-contained so that the backend
        can run without any external database. During training you will replace
        this with code that builds the vocabulary from MIMIC-III/IV or other
        EHR data and loads a trained checkpoint from disk.
        """
        import csv
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / "dataset" / "text" / "medicine.csv"

        meds: List[str] = []
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("Drug_Name") or "").strip()
                    if not name:
                        continue
                    if name not in meds:
                        meds.append(name)
        else:
            # Fallback list if the CSV is missing.
            meds = ["DERM_DRUG_A", "DERM_DRUG_B", "DERM_DRUG_C", "DERM_DRUG_D"]

        # Simple text vocabulary. We reserve:
        #   0: [PAD], 1: [UNK], 2: [CLS]
        base_tokens = ["[PAD]", "[UNK]", "[CLS]"]
        extra_tokens: List[str] = []

        # Optionally derive additional tokens from the medicine CSV reasons.
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in ("Reason", "Description"):
                        text = (row.get(key) or "").upper()
                        for token in text.replace("-", " ").replace("/", " ").split():
                            if not token.isalpha():
                                continue
                            if token not in extra_tokens and token not in base_tokens:
                                extra_tokens.append(token)
                            if len(extra_tokens) >= max_vocab_size - len(base_tokens):
                                break
                        if len(extra_tokens) >= max_vocab_size - len(base_tokens):
                            break
                    if len(extra_tokens) >= max_vocab_size - len(base_tokens):
                        break

        vocab_tokens = base_tokens + extra_tokens
        vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(vocab_tokens)}

        model = SimplePMRSModel(
            vocab_size=len(vocab_tokens),
            num_meds=len(meds),
            text_dim=128,
            image_dim=128,
            hidden_dim=64,
            max_len=256,
        )

        # Use a fixed random seed so the demo behaves deterministically.
        torch.manual_seed(42)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model, vocab, meds


@dataclass(frozen=True)
class TokenisedInput:
    """Small helper container used by the service layer."""

    token_ids: Tensor
    padding_mask: Tensor
