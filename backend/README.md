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

The API root will be available at: `http://localhost:8000`

Interactive docs (FastAPI Swagger UI):

- Swagger: `http://localhost:8000/docs`
- ReDoc:   `http://localhost:8000/redoc`
