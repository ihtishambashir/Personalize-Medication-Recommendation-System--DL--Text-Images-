# Backend Understanding + Vision Transformer Pattern Module

This document explains:

1. The full backend code structure.
2. The Vision Transformer implementation in `app/models/vision_transformer.py`.
3. The new upload module that shows how image patterns are recognized.

## 1) Full Backend Structure (Quick Map)

- `app/main.py`
  - Creates the FastAPI app, CORS config, and mounts routers.
- `app/api/routes.py`
  - Text recommendation endpoint (`/api/recommend`).
- `app/api/vision.py`
  - Vision endpoints:
    - `/api/vision/skin-diagnosis`
    - `/api/vision/skin-pattern-analysis` (new).
- `app/services/recommendation.py`
  - Loads text model artifacts (if present), falls back to demo multimodal model, adds DDI warnings.
- `app/services/vision.py`
  - Loads ViT checkpoint once, preprocesses uploaded images, returns top-k predictions.
- `app/services/vision_pattern.py` (new)
  - Runs explainable pattern analysis for uploaded images (patch details + salient regions).
- `app/models/text_transformer.py`
  - Vocabulary helper + compact text Transformer classifier + checkpoint helpers.
- `app/models/vision_transformer.py`
  - ViT model creation, save/load checkpoint, and ViT architecture helper utilities.
- `app/models/ddi_graph.py`
  - Toy in-memory drug-drug interaction checker.
- `app/training/train_text_transformer.py`
  - Training pipeline for text model.
- `app/training/train_vit_skin.py`
  - Training pipeline for ViT skin classifier.

## 2) Vision Transformer Code (Human-Written Explanation)

File: `app/models/vision_transformer.py`

- `create_vit_model(...)`
  - Builds a `torchvision.vit_b_16` model.
  - Replaces the classifier head with:
    - `LayerNorm`
    - optional `Dropout`
    - final `Linear(in_features -> num_classes)`.
  - This keeps backbone behavior but adapts output classes to your dataset.

- `save_vit_checkpoint(...)`
  - Saves:
    - model weights (`vision_transformer.pt`)
    - class mapping (`classes.json`)
    - training/inference config (inside checkpoint).

- `load_vit_checkpoint(...)`
  - Loads both files.
  - Reconstructs the same model shape from class count and config.
  - Restores weights and switches model to eval mode.
  - Returns `VisionArtifacts(model, idx_to_class, config)`.

- Helper utilities
  - `get_vit_patch_size(...)`
  - `get_vit_encoder_depth(...)`
  - `get_vit_num_heads(...)`
  - These are used by the new pattern-analysis module to explain how ViT processes image patches.

## 3) Vision Transformer Code (AI-Style Structured Explanation)

1. Input checkpoint path.
2. Validate required files (`vision_transformer.pt`, `classes.json`).
3. Parse class mapping and build index-to-label list.
4. Build ViT-B/16 with a custom classification head.
5. Load checkpoint state dict into model.
6. Send model to selected device (`cpu`/`cuda`) and set `eval()`.
7. Return artifacts for inference service consumption.

## 4) New Upload Module for Pattern Recognition

### New service module

File: `app/services/vision_pattern.py`

Main function:

- `analyze_skin_pattern(image_bytes, top_k=3)`
  - Uses the same loaded model + transforms from `app/services/vision.py`.
  - Predicts top-k classes.
  - Extracts ViT structural info:
    - input image size
    - patch size
    - patch grid rows/cols
    - number of transformer layers
    - attention head count
  - Computes gradient-based saliency hotspots and maps them to patch coordinates.
  - Returns a concise summary of where the strongest evidence was found.

### New API endpoint

File: `app/api/vision.py`

- `POST /api/vision/skin-pattern-analysis`
  - Input: uploaded image (`multipart/form-data`, field name `file`).
  - Output:
    - `predicted_label`, `confidence`, `top_k`
    - `model_info` (ViT patch/layer details)
    - `recognition_steps`
    - `salient_patches`
    - `pattern_summary`
    - `disclaimer`
- `POST /api/vision/skin-pattern-analysis/image`
  - Input: uploaded image (`multipart/form-data`, field name `file`).
  - Output: `image/png` diagnostic overlay with:
    - color heatmap for model focus intensity
    - highlighted top patch boxes
    - predicted class + confidence text strip

### New response schemas

File: `app/schemas/vision.py`

- `VisionPatternResponse`
- `VisionPatternModelInfo`
- `VisionSalientPatch`

## 5) How ViT Recognizes Patterns in This Backend

1. Uploaded image is decoded to RGB.
2. Image is resized/cropped to fixed model size (default 224x224).
3. Pixel normalization is applied (ImageNet mean/std by default).
4. Image is split into fixed-size patches (ViT-B/16 -> 16x16 patches).
5. Patches are projected into tokens and passed through Transformer encoder layers.
6. Classification head predicts class logits, softmax gives probabilities.
7. Saliency module backpropagates top-class logit to input and reports top influential patch locations.

## 6) Example Request

```bash
curl -X POST "http://localhost:8000/api/vision/skin-pattern-analysis?top_k=3" ^
  -F "file=@\"dataset/skin images/test/nv/ISIC_0024306.jpg\""
```

```bash
curl -X POST "http://localhost:8000/api/vision/skin-pattern-analysis/image?top_k=3" ^
  -F "file=@\"dataset/skin images/test/nv/ISIC_0024306.jpg\"" ^
  --output skin_pattern_overlay.png
```

## 7) Important Note

This is research code for thesis/demo use only.  
It is not medically validated and must not be used in real clinical decision-making.
