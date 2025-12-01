# PMRS Demo Backend (FastAPI + PyTorch)

This backend is a **research-only** demo API for a Personalized Medication Recommendation System (PMRS)
as described in your thesis draft. It exposes a simple HTTP endpoint that accepts EHR-like text data and
returns dummy medication suggestions with a toy DDI safety filter.

> ⚠️ **Critical disclaimer**  
> This code is **not** a medical device, has **no clinical validation**, and must **never** be used for
> real diagnosis or treatment decisions. It is for academic experimentation and UI demonstration only.

## How to run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
## How to train transformer model

```bash
python -m app.training.train_text_transformer --data-path dataset/text/medicine.csv --output-dir trained_models/text_transformer --epochs 10 --batch-size 64 --max-len 64

```

The API root will be available at: `http://localhost:8000`

Interactive docs (FastAPI Swagger UI):

- Swagger: `http://localhost:8000/docs`

## Input 

```JSON

{
  "diagnoses": ["Acne vulgaris"],
  "symptoms": ["pustules on face", "oily skin"],
  "notes": "Teenager with moderate acne, otherwise healthy.",
  "current_medications": ["Paracetamol"]
}

```
- That text (diagnoses + symptoms + notes) is what the Transformer model “reads”. It predicts a Reason (label from your medicine csv), and then the backend maps that Reason → medicines from the CSV and returns suggestions.

## Output 

```JSON

{
  "suggestions": [
    {
      "drug_code": "isotretinoin",
      "name": "Isotretinoin",
      "score": 0.87,
      "warnings": []
    },
    {
      "drug_code": "doxycycline",
      "name": "Doxycycline",
      "score": 0.65,
      "warnings": []
    }
  ],
  "ddi_warnings": [
    "No known interactions in this toy graph."
  ],
  "disclaimer": "This is a research demo only ... (etc)"
}
```


- ReDoc:   `http://localhost:8000/redoc`
