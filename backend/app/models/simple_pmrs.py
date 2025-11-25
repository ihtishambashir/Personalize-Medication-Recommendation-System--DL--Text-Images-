from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class SimplePMRSModel(nn.Module):
    """A small PyTorch model that mimics the idea of EHR encoding + medication scoring.

    This is **not** a clinically meaningful model. It is just a compact, fully
    self-contained example that you can extend in your thesis to match your
    actual architecture (sequence encoder, image encoder, graph module, etc.).
    """

    def __init__(self, vocab_size: int, num_meds: int, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_meds)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LongTensor of shape (batch, seq_len) with token ids.

        Returns:
            probs: FloatTensor of shape (batch, num_meds) with scores in [0, 1].
        """
        emb = self.embedding(x)
        _, h_n = self.encoder(emb)
        h_last = h_n[-1]
        logits = self.classifier(h_last)
        probs = self.sigmoid(logits)
        return probs

    @staticmethod
    def build_demo() -> tuple["SimplePMRSModel", Dict[str, int], List[str]]:
        """Create a tiny demo model with a toy vocabulary and medication list.

        Returns:
            model, vocab, meds
        """
        # Toy vocab: in practice you would build this from EHR codes.
        vocab_tokens = [
            "PAD",  # 0: padding
            "DX_DIABETES",
            "DX_HYPERTENSION",
            "DX_RASH",
            "PROC_BLOOD_TEST",
            "MED_METFORMIN",
            "MED_ACE_INHIBITOR",
        ]
        vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(vocab_tokens)}

        # Toy medication list (demo only).
        meds = [
            "DRUG_A",
            "DRUG_B",
            "DRUG_C",
            "DRUG_D",
        ]
        model = SimplePMRSModel(vocab_size=len(vocab_tokens), num_meds=len(meds), hidden_dim=32)
        return model, vocab, meds
