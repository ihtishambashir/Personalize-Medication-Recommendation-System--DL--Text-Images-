# Vision Transformer (ViT) and Text Transformer

Professional technical documentation for the backend implementation in this repository.

## 1. Purpose and Scope

This document describes:

1. Where the ViT and text Transformer implementations come from.
2. How each model is built and used in this backend.
3. Training and inference flow, including generated artifacts.
4. Functional boundaries, assumptions, and known limitations.

Scope is limited to code under `backend/app` and `backend/trained_models`.

## 2. Provenance: Where the Models Come From

## 2.1 Vision Transformer (ViT)

- Primary implementation source:
  - `torchvision.models.vit_b_16`
- Referenced in code:
  - `app/models/vision_transformer.py`
  - function: `create_vit_model(...)`

Important details:

- Backbone is imported from `torchvision` (not custom-written from scratch).
- Optional pretrained initialization is available through:
  - `models.ViT_B_16_Weights.IMAGENET1K_V1`
- In this repository, pretrained weights are used during training only if the CLI flag `--pretrained` is supplied in:
  - `app/training/train_vit_skin.py`

## 2.2 Text Transformer

- Primary implementation source:
  - `torch.nn.TransformerEncoderLayer`
  - `torch.nn.TransformerEncoder`
- Referenced in code:
  - `app/models/text_transformer.py`
  - class: `TextTransformerClassifier`

Important details:

- This is a custom compact encoder classifier built directly with PyTorch primitives.
- It is not a Hugging Face pretrained language model.
- Tokenization/vocabulary is custom (`Vocab` class), based on lowercase whitespace splitting.

## 3. Dependency and Runtime Requirements

Model dependencies are declared in `requirements.txt`:

- `torch`
- `torchvision`
- `numpy`
- `pillow`
- `pandas`
- `matplotlib`
- `fastapi`, `uvicorn[standard]`

The backend can run inference on CPU or CUDA based on availability:

- Vision service device selection:
  - `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
  - in `app/services/vision.py`
- Recommendation text inference currently uses CPU by default in:
  - `app/services/recommendation.py`

## 4. Vision Transformer Architecture and Functionality

## 4.1 Model Construction

File: `app/models/vision_transformer.py`

- `create_vit_model(num_classes, pretrained=False, dropout=0.1)`
  - Loads `vit_b_16` backbone from torchvision.
  - Replaces classification head with:
    - `LayerNorm(in_features)`
    - optional `Dropout(dropout)`
    - `Linear(in_features, num_classes)`
  - This ensures output logits match project class count.

## 4.2 Artifact Persistence

- `save_vit_checkpoint(...)` writes:
  - `vision_transformer.pt` (state dict + config)
  - `classes.json` (class name -> class index)

- `load_vit_checkpoint(...)` reads those files and reconstructs model for inference.

Additional model-introspection helpers:

- `get_vit_patch_size(model)`
- `get_vit_encoder_depth(model)`
- `get_vit_num_heads(model)`

These are used by the explainability/pattern-analysis service.

## 4.3 Training Pipeline

File: `app/training/train_vit_skin.py`

Data expectation:

- `dataset/skin images/train/<class>/*.jpg`
- `dataset/skin images/val/<class>/*.jpg`
- `dataset/skin images/test/<class>/*.jpg`

Training behavior:

1. Data augmentations for training split.
2. Center-crop evaluation pipeline for validation/test.
3. Loss: `CrossEntropyLoss`.
4. Optimizer: `AdamW`.
5. LR scheduler: `CosineAnnealingLR`.
6. Best validation checkpoint selected and then evaluated on test split.

Training outputs:

- `trained_models/vision_transformer/vision_transformer.pt`
- `trained_models/vision_transformer/classes.json`
- `trained_models/vision_transformer/metrics.json`
- `trained_models/vision_transformer/accuracy_curve.png`
- `trained_models/vision_transformer/loss_curve.png`

## 4.4 Inference and API Usage

Files:

- `app/services/vision.py`
- `app/api/vision.py`

Endpoints:

- `POST /api/vision/skin-diagnosis`
  - Returns top-k predicted labels with softmax confidence.

- `POST /api/vision/skin-pattern-analysis`
  - Returns prediction + explainability metadata (patch grid, salient patches, processing steps).

- `POST /api/vision/skin-pattern-analysis/image`
  - Returns `image/png` color overlay highlighting diagnosis-focused regions.

Returned image overlay includes:

- per-pixel saliency heatmap blend,
- top salient patch boxes,
- predicted label + confidence text strip.

## 5. Text Transformer Architecture and Functionality

## 5.1 Vocabulary and Tokenization

File: `app/models/text_transformer.py`

- `Vocab` class:
  - Lowercase + whitespace tokenization.
  - Reserved tokens:
    - `<pad>` id `0`
    - `<unk>` id `1`
  - Can serialize/deserialize to JSON.

## 5.2 Model Definition

- `TextTransformerClassifier`
  - Embedding layer (`nn.Embedding`).
  - Sinusoidal positional encoding (`PositionalEncoding`).
  - Transformer encoder stack (`nn.TransformerEncoder`).
  - LayerNorm + mean pooling (mask-aware) + linear classifier.

Input/Output contract:

- Input:
  - `token_ids` shape `(batch, seq_len)`
  - optional `padding_mask` shape `(batch, seq_len)`
- Output:
  - logits shape `(batch, num_classes)`

## 5.3 Training Pipeline

File: `app/training/train_text_transformer.py`

Dataset source:

- `dataset/text/medicine.csv`

Expected CSV columns:

- `Reason`
- `Description`
- `Drug_Name`

Training behavior:

1. Build text input from `Description` (fallback to `Drug_Name`).
2. Map `Reason` to classification labels.
3. Train/validation split.
4. Optimizer: `Adam`.
5. Loss: `CrossEntropyLoss`.

Training outputs:

- `trained_models/text_transformer/text_transformer.pt`
- `trained_models/text_transformer/vocab.json`
- `trained_models/text_transformer/labels.json`
- `trained_models/text_transformer/train_acc.npy`
- `trained_models/text_transformer/val_acc.npy`
- optional `trained_models/text_transformer/accuracy_curve.png`

## 5.4 Inference in Recommendation Service

File: `app/services/recommendation.py`

Operational flow:

1. Load text model artifacts (if present).
2. Build request text from diagnoses/symptoms/notes.
3. Predict top reason classes.
4. Map predicted reasons to medication names using `medicine.csv`.
5. Return medication suggestions with scores.
6. Run toy DDI checks via `DDIGraph`.

Fallback logic:

- If trained text artifacts are unavailable or mapping fails, service falls back to demo multimodal model in `app/models/simple_pmrs.py`.

## 6. End-to-End Functional Architecture

High-level flow in production API:

1. Client submits text and/or image payload.
2. API route validates request.
3. Service layer loads/calls model artifacts.
4. Model returns logits -> probabilities.
5. Service formats output (predictions/explanations/warnings/disclaimer).
6. API returns JSON or PNG response.

## 7. Explainability Notes (Current Implementation)

Explainability currently uses gradient-based saliency:

1. Backpropagate top-class logit to input image.
2. Aggregate gradient magnitudes across channels.
3. Normalize and smooth to create saliency map.
4. Convert to patch-level importance for top hotspot boxes.
5. Render overlay heatmap (for `/skin-pattern-analysis/image`).

This provides directional interpretability, not clinical-grade causality.

## 8. Limitations and Safety Position

- This repository is research/demo code.
- Outputs are not clinically validated.
- DDI graph is explicitly toy/demo logic.
- Vision saliency is explanatory but not diagnostic evidence.

Clinical deployment requires:

1. regulated validation,
2. external safety auditing,
3. domain expert review,
4. robust data governance and post-deployment monitoring.

## 9. Reproducibility Checklist

1. Install dependencies from `requirements.txt`.
2. Ensure dataset folders/files follow expected structure.
3. Train text model and ViT model using provided scripts.
4. Confirm artifacts exist under `trained_models/...`.
5. Start API and verify `/docs`.
6. Test endpoints with known samples and track output artifacts.

## 10. File Reference Index

- ViT model factory/loading:
  - `app/models/vision_transformer.py`
- ViT training:
  - `app/training/train_vit_skin.py`
- Vision inference service:
  - `app/services/vision.py`
- Vision explainability + overlay:
  - `app/services/vision_pattern.py`
- Vision API routes:
  - `app/api/vision.py`
- Text model:
  - `app/models/text_transformer.py`
- Text training:
  - `app/training/train_text_transformer.py`
- Recommendation orchestration:
  - `app/services/recommendation.py`

