from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
from PIL import Image

from app.models.ddi_graph import DDIGraph
from app.models.simple_pmrs import SimplePMRSModel, TokenisedInput
from app.models.text_transformer import Vocab, load_inference_artifacts
from app.schemas.medication import (
    MedicationSuggestion,
    RecommendationRequest,
    RecommendationResponse,
)

# ---------------------------------------------------------------------------
# Global resources – created once at import time
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEVICE = torch.device("cpu")

# 1) Try to load the text Transformer trained on dataset/text/medicine.csv.
_TEXT_MODEL_DIR = _PROJECT_ROOT / "trained_models" / "text_transformer"

_text_model = None
_text_vocab: Vocab | None = None
_idx_to_reason: List[str] = []
_text_config: Dict[str, object] = {}

try:
    (
        _text_model,
        _text_vocab,
        _idx_to_reason,
        _text_config,
    ) = load_inference_artifacts(_TEXT_MODEL_DIR, device=_DEVICE)
except FileNotFoundError:
    # The user has not trained the Transformer yet; the API will fall back
    # to the original demo model defined in simple_pmrs.py.
    _text_model = None
    _text_vocab = None
    _idx_to_reason = []
    _text_config = {}

# 2) Load the medicine.csv file and build a simple mapping:
#       Reason -> list of candidate medications.
_reason_to_meds: Dict[str, List[str]] = {}
_all_med_names: List[str] = []

_MEDICINE_CSV = _PROJECT_ROOT / "dataset" / "text" / "medicine.csv"
if _MEDICINE_CSV.exists():
    try:
        df = pd.read_csv(_MEDICINE_CSV)
        if {"Reason", "Drug_Name"} <= set(df.columns):
            df = df.dropna(subset=["Reason", "Drug_Name"])
            df["Reason"] = df["Reason"].astype(str).str.strip()
            df["Drug_Name"] = df["Drug_Name"].astype(str).str.strip()

            for reason, group in df.groupby("Reason"):
                meds = sorted({name for name in group["Drug_Name"].tolist() if name})
                if meds:
                    _reason_to_meds[reason] = meds
                    _all_med_names.extend(meds)

            _all_med_names = sorted(set(_all_med_names))
    except Exception:
        # If anything goes wrong we simply keep the mapping empty and fall
        # back to the demo model.
        _reason_to_meds = {}
        _all_med_names = []

# 3) Fallback demo model (Transformer + CNN) from simple_pmrs.py
#    – used only when the new text model is absent.
_model: SimplePMRSModel | None = None
_vocab: Dict[str, int] | None = None  # from TokenisedInput
_meds: List[str] = []

if _text_model is None:
    # Build the original demo artefacts.
    _model, _vocab, _meds = SimplePMRSModel.build_demo()
    if not _all_med_names:
        _all_med_names = _meds

# 4) DDI graph uses whatever medication names we have available.
_ddi_graph = DDIGraph.build_demo(_all_med_names or _meds)

# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------


def _normalise_text(text: str) -> str:
    return text.strip().upper()


def _tokenise_to_ids(tokens: List[str]) -> TokenisedInput:
    """Convert a list of text snippets into a TokenisedInput for the demo model.

    We keep the logic intentionally simple: concatenate snippets, do a crude
    upper‑case split on whitespace and map tokens through the demo vocab.
    No padding is performed here – the sequence is processed as is.
    """
    if _vocab is None:
        raise RuntimeError("Demo vocabulary is not initialised.")

    joined = " [SEP] ".join(_normalise_text(t) for t in tokens if t)
    words = joined.split()

    token_ids: List[int] = []
    for w in words:
        token_ids.append(int(_vocab.get(w, 0)))

    if not token_ids:
        token_ids.append(0)

    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    padding_mask = torch.zeros_like(token_tensor, dtype=torch.bool)

    return TokenisedInput(token_ids=token_tensor, padding_mask=padding_mask)
def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalised CHW tensor."""
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, C)
    if arr.ndim == 2:  # grayscale
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
    return torch.from_numpy(arr)


def _load_images(image_paths: List[str]) -> torch.Tensor | None:
    """Load and preprocess images if any paths are provided.

    The function is deliberately defensive: if an image cannot be read for any
    reason we simply ignore it instead of failing the whole request.
    """
    if not image_paths:
        return None

    project_root = Path(__file__).resolve().parents[2]

    images: List[torch.Tensor] = []
    for rel_path in image_paths:
        full_path = (project_root / rel_path).resolve()
        if not full_path.exists():
            continue
        try:
            img = Image.open(full_path).convert("RGB")
            images.append(_pil_to_tensor(img))
        except Exception:
            # Ignore unreadable files – the backend should remain robust.
            continue

    if not images:
        return None

    batch = torch.stack(images, dim=0)
    return batch


# ---------------------------------------------------------------------------
# Core recommendation logic
# ---------------------------------------------------------------------------


def _build_request_text(request: RecommendationRequest) -> str:
    """Concatenate the textual fields of the request into a single string."""
    parts: List[str] = []
    parts.extend(request.diagnoses or [])
    parts.extend(request.symptoms or [])
    if request.notes:
        parts.append(request.notes)
    if not parts:
        return "no diagnosis provided"
    return " ".join(parts)


def _recommend_with_trained_text_model(
    request: RecommendationRequest,
) -> List[MedicationSuggestion]:
    """Use the Transformer trained on medicine.csv to suggest medications.

    The model predicts a probability distribution over the high-level
    'Reason' labels, which we then map to concrete medications from the CSV.
    """
    assert _text_model is not None and _text_vocab is not None and _idx_to_reason

    text = _build_request_text(request)
    max_len = int(_text_config.get("max_len", 64))

    # Encode the single example and build a padding mask.
    token_ids = torch.tensor(
        [_text_vocab.encode(text, max_len=max_len)],
        dtype=torch.long,
        device=_DEVICE,
    )
    padding_mask = token_ids.eq(_text_vocab.pad_id)

    with torch.no_grad():
        logits = _text_model(token_ids, padding_mask=padding_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_reasons,)

    # Take the top few reasons; we will then expand them into medications.
    num_reasons = len(_idx_to_reason)
    top_k_reasons = min(5, num_reasons)
    top_probs, top_indices = torch.topk(probs, top_k_reasons)

    scores_by_med: Dict[str, float] = {}
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        reason = _idx_to_reason[idx]
        meds_for_reason = _reason_to_meds.get(reason, [])
        for name in meds_for_reason:
            # Keep the best score for each medication name.
            scores_by_med[name] = max(scores_by_med.get(name, 0.0), float(prob))

    if not scores_by_med:
        # If we cannot map reasons to medications (e.g. CSV missing), bail out
        # to the demo model.
        return []

    # Sort medications by score and respect the max_suggestions limit.
    items: List[Tuple[str, float]] = sorted(
        scores_by_med.items(), key=lambda kv: kv[1], reverse=True
    )
    k = int(request.max_suggestions)
    k = max(1, min(k, len(items)))
    top_items = items[:k]

    suggestions: List[MedicationSuggestion] = []
    for idx, (name, score) in enumerate(top_items):
        suggestions.append(
            MedicationSuggestion(
                drug_code=f"MED_{idx}",
                name=name,
                score=float(score),
                warnings=[],
            )
        )

    return suggestions


def _recommend_with_demo_model(
    request: RecommendationRequest,
) -> List[MedicationSuggestion]:
    """Fallback: use the original SimplePMRSModel demo implementation."""
    if _model is None or _vocab is None or not _meds:
        return []

    # 1) Prepare textual input (diagnoses, symptoms, notes).
    text_fields: List[str] = []
    text_fields.extend(request.diagnoses or [])
    text_fields.extend(request.symptoms or [])
    if request.notes:
        text_fields.append(request.notes)

    if not text_fields:
        text_fields.append("NO_DIAGNOSIS_PROVIDED")

    tokenised = _tokenise_to_ids(text_fields)

    # 2) Prepare images (optional).
    image_batch = _load_images(request.image_paths)

    # 3) Run the model.
    _model.eval()
    with torch.no_grad():
        logits = _model(
            tokenised.token_ids.to(_DEVICE),
            padding_mask=tokenised.padding_mask.to(_DEVICE),
            images=image_batch.to(_DEVICE) if image_batch is not None else None,
        )
        scores = torch.sigmoid(logits).squeeze(0)  # (num_meds,)

    # 4) Pick top-k medications.
    k = int(request.max_suggestions)
    k = max(1, min(k, len(_meds)))
    top_scores, top_indices = torch.topk(scores, k)

    suggestions: List[MedicationSuggestion] = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        name = _meds[idx]
        suggestions.append(
            MedicationSuggestion(
                drug_code=f"MED_{idx}",
                name=name,
                score=float(score),
                warnings=[],
            )
        )

    return suggestions


def recommend_medications(request: RecommendationRequest) -> RecommendationResponse:
    """Entry point used by the FastAPI router.

    The function first tries to use the Transformer model that you can train
    on dataset/text/medicine.csv via the CLI script:

        python -m app.training.train_text_transformer --epochs 10

    If that model has not been trained yet, we fall back to the original
    demo model defined in app.models.simple_pmrs.
    """
    # First choice: the trained text Transformer, if available.
    suggestions: List[MedicationSuggestion]
    if _text_model is not None and _text_vocab is not None and _idx_to_reason:
        suggestions = _recommend_with_trained_text_model(request)
    else:
        suggestions = []

    # Fallback to the demo model if needed.
    if not suggestions:
        demo_suggestions = _recommend_with_demo_model(request)
        if demo_suggestions:
            suggestions = demo_suggestions

    # 5) Run the toy DDI checker on the suggested combination plus current meds.
    suggested_names = [s.name for s in suggestions]
    all_for_check = list({*suggested_names, *(request.current_medications or [])})
    ddi_warnings = _ddi_graph.check_combination(all_for_check)

    disclaimer = (
        "This PMRS backend is a research prototype only. "
        "The recommendations are generated by unvalidated machine‑learning models "
        "(a small Transformer trained on medicine.csv and/or a demo multimodal model) "
        "together with a toy DDI graph. "
        "They must not be used for any real diagnosis, treatment, or clinical decision‑making."
    )

    return RecommendationResponse(
        suggestions=suggestions,
        ddi_warnings=ddi_warnings,
        disclaimer=disclaimer,
    )
