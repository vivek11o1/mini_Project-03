# Loan Approval Prediction (FastAPI)

Predicts whether a loan application will be **Approved** or **Rejected** using a
**Random Forest Classifier** (20 trees, entropy criterion). The trained artifacts
`model.pkl`, `scaler.pkl`, and `encoder.pkl` are already included.

The service is exposed through **FastAPI** with Swagger UI at `http://localhost:8000/docs`.
No frontend is required.

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | FastAPI service (endpoints + Swagger UI) |
| `model.pkl` | Trained Random Forest classifier |
| `scaler.pkl` | StandardScaler fitted on the 11 input features |
| `encoder.pkl` | LabelEncoder for `self_employed` (No=0, Yes=1) |
| `Loan approval Project.ipynb` | Original notebook: EDA, training, evaluation (~97% accuracy) |
| `test_api.py` | Smoke test for the API |
| `Dockerfile`, `.dockerignore` | Docker/Render deployment config |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> `scikit-learn` is pinned to **1.5.1** and `numpy` to **1.26.4** — the versions used
> to create the committed `.pkl` files. Newer sklearn versions warn about
> incompatible unpickling and may give different results.

## Run the API

```bash
uvicorn app:app --reload
```

Open **http://localhost:8000/docs** for Swagger UI.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check, model name, feature count, classes |
| POST | `/predict` | Predict one loan application |
| POST | `/predict/batch` | Predict several applications |
| GET | `/docs` | Swagger UI |

The `/predict` request body is a JSON object with these 11 fields:

```json
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 9600000,
  "loan_amount": 29900000,
  "loan_term": 12,
  "cibil_score": 778,
  "residential_assets_value": 2400000,
  "commercial_assets_value": 17600000,
  "luxury_assets_value": 22700000,
  "bank_asset_value": 8000000
}
```

Example response:

```json
{
  "status": "Approved",
  "approved": true,
  "confidence": 1.0,
  "probabilities": {"Rejected": 0.0, "Approved": 1.0}
}
```

A real rejected row from the dataset (low CIBIL score) returns
`"status": "Rejected"` with confidence `1.0`. Both examples are used in `test_api.py`.

### Encoding rule (important)

The original notebook used `LabelEncoder`, which produced:

- `education`: **Graduate = 0**, **Not Graduate = 1**
- `self_employed`: **No = 0**, **Yes = 1**

The original `app.py` had the education mapping reversed; this version uses the
correct mapping so predictions match the trained model.

## Test

```bash
python test_api.py
```

## Docker / deployment

```bash
docker build -t loan-approval .
docker run -p 8000:8000 loan-approval
```

The container reads the `PORT` environment variable (Render injects `PORT=10000`
by default) and falls back to `8000` locally. It binds to `0.0.0.0`, as required
by Render. Use **Docker** as the runtime, health check path `/health`, and an
initial health check delay of ~15s.
