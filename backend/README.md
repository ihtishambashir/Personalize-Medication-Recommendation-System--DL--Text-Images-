# PMRS Demo Backend (FastAPI + PyTorch)

Research-only FastAPI backend for the thesis prototype. It offers:

- Text-based medication suggestions with a small Transformer.
- Vision Transformer (ViT) classifier for skin images (train/val/test folders).
- Toy drug-drug interaction warnings to illustrate safety checks.

> WARNING  
> This project is not a medical device, has no clinical validation, and must never be used for diagnosis or treatment. It is only for academic experimentation and UI demos.

## Quick start

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # on Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: `http://localhost:8000/docs`  
ReDoc: `http://localhost:8000/redoc`

## Training

### Text Transformer (EHR text)

Trains on `dataset/text/medicine.csv` and writes artifacts to `trained_models/text_transformer`:

```bash
python -m app.training.train_text_transformer --data-path dataset/text/medicine.csv --output-dir trained_models/text_transformer --epochs 10 --batch-size 64 --max-len 64
```

Artifacts: `text_transformer.pt`, `vocab.json`, `labels.json`, accuracy curves (PNG + NPY).

### Vision Transformer (skin images)

Expected layout: `dataset/skin images/{train,val,test}/{class_name}/*.jpg`

```bash
python -m app.training.train_vit_skin --data-root "dataset/skin images" --output-dir trained_models/vision_transformer --epochs 15 --batch-size 16 --image-size 224 --pretrained --device cuda
```

Outputs in `trained_models/vision_transformer`:

- `vision_transformer.pt`, `classes.json`, `metrics.json`, config in the checkpoint.
- Accuracy and loss plots: `accuracy_curve.png`, `loss_curve.png`.
- Test metrics recorded in `metrics.json` with per-class accuracy.

## API endpoints

### Medication recommendation (text)

`POST /api/recommend`

```json
{
  "diagnoses": ["Acne vulgaris"],
  "symptoms": ["pustules on face", "oily skin"],
  "notes": "Teenager with moderate acne, otherwise healthy.",
  "current_medications": ["Paracetamol"],
  "max_suggestions": 5
}
```

Response: suggested medications with scores, toy DDI warnings, and a disclaimer.

### Skin diagnosis (ViT)

`POST /api/vision/skin-diagnosis?top_k=3`  
Payload: multipart form with file field `file` containing a JPG/PNG image.

```bash
curl -X POST "http://localhost:8000/api/vision/skin-diagnosis?top_k=3" ^
  -F "file=@\"dataset/skin images/test/nv/ISIC_0024306.jpg\""
```

Response: top-k class labels with softmax scores plus a research-only disclaimer. The endpoint returns 503 if the ViT checkpoint is missing.

### Skin pattern analysis (ViT explainability)

`POST /api/vision/skin-pattern-analysis?top_k=3`  
Payload: multipart form with file field `file` containing a JPG/PNG image.

```bash
curl -X POST "http://localhost:8000/api/vision/skin-pattern-analysis?top_k=3" ^
  -F "file=@\"dataset/skin images/test/nv/ISIC_0024306.jpg\""
```

Response includes:

- top-k predictions
- ViT architecture metadata (patch size, patch grid, layers, heads)
- recognition steps
- salient patches (most influential image regions)
- research-only disclaimer

### Skin pattern colored output image

`POST /api/vision/skin-pattern-analysis/image?top_k=3`  
Payload: multipart form with file field `file` containing a JPG/PNG image.

```bash
curl -X POST "http://localhost:8000/api/vision/skin-pattern-analysis/image?top_k=3" ^
  -F "file=@\"dataset/skin images/test/nv/ISIC_0024306.jpg\"" ^
  --output skin_pattern_overlay.png
```

This returns a PNG with a realistic color heatmap over the lesion area and highlighted patch boxes where the model focused most.

## Documentation

- `docs/VISION_TRANSFORMER_CODE_AND_PATTERN_MODULE.md`: full backend code understanding plus detailed Vision Transformer and pattern module explanation (human-style + AI-style).
- `docs/VIT_AND_TEXT_TRANSFORMER_PROFESSIONAL_DOCUMENTATION.md`: professional technical documentation for ViT + text Transformer provenance, architecture, training, inference, and functionality.

---

This codebase is for research and educational purposes only; do not use it for any clinical workflow.
