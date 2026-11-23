"""FastAPI service for Loan Approval Prediction (Swagger UI at /docs).

Run locally:
    uvicorn app:app --reload
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent


def _load_artifact(filename: str):
    path = BASE_DIR / filename
    if not path.exists():
        raise RuntimeError(f"Missing artifact '{filename}' in {BASE_DIR}")
    with path.open("rb") as file:
        return pickle.load(file)


model = _load_artifact("model.pkl")
scaler = _load_artifact("scaler.pkl")
encoder = _load_artifact("encoder.pkl")  # LabelEncoder for self_employed: No=0, Yes=1

# The training notebook used LabelEncoder on education: 'Graduate'=0, 'Not Graduate'=1.
EDUCATION_MAP = {"Graduate": 0, "Not Graduate": 1}

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predict loan approval using a Random Forest classifier trained on 11 tabular features.",
    version="1.0.0",
)


class LoanApplication(BaseModel):
    no_of_dependents: int = Field(..., ge=0, le=10, description="Number of dependents")
    education: Literal["Graduate", "Not Graduate"]
    self_employed: Literal["No", "Yes"]
    income_annum: float = Field(..., ge=0, description="Annual income")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_term: int = Field(..., gt=0, description="Loan term in months")
    cibil_score: int = Field(..., ge=300, le=900, description="Credit score")
    residential_assets_value: float = Field(..., ge=0)
    commercial_assets_value: float = Field(..., ge=0)
    luxury_assets_value: float = Field(..., ge=0)
    bank_asset_value: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "bank_asset_value": 8000000,
            }
        }
    }


class BatchRequest(BaseModel):
    applications: List[LoanApplication] = Field(..., min_length=1)


class Prediction(BaseModel):
    status: str
    approved: bool
    confidence: float
    probabilities: Dict[str, float]


class BatchResponse(BaseModel):
    predictions: List[Prediction]


def _status(value: int) -> str:
    return "Approved" if int(value) == 1 else "Rejected"


def _predict(application: LoanApplication) -> Prediction:
    features = np.array(
        [[
            application.no_of_dependents,
            EDUCATION_MAP[application.education],
            int(encoder.transform([application.self_employed])[0]),
            application.income_annum,
            application.loan_amount,
            application.loan_term,
            application.cibil_score,
            application.residential_assets_value,
            application.commercial_assets_value,
            application.luxury_assets_value,
            application.bank_asset_value,
        ]],
        dtype=float,
    )
    scaled = scaler.transform(features)
    predicted = int(model.predict(scaled)[0])
    probabilities = model.predict_proba(scaled)[0]
    scores = {
        _status(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    status = _status(predicted)
    return Prediction(
        status=status,
        approved=predicted == 1,
        confidence=scores[status],
        probabilities=scores,
    )


@app.get("/", tags=["info"], summary="API information")
def root():
    return {
        "name": "Loan Approval Prediction API",
        "model": "Random Forest Classifier",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["info"], summary="Health check")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model": type(model).__name__,
        "n_features": int(scaler.n_features_in_),
        "classes": [_status(label) for label in model.classes_],
    }


@app.post("/predict", response_model=Prediction, tags=["prediction"], summary="Predict one loan")
def predict(application: LoanApplication):
    """Return Approved/Rejected status, confidence and class probabilities."""
    return _predict(application)


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Predict several loans",
)
def predict_batch(batch: BatchRequest):
    """Classify multiple applications in one request."""
    return BatchResponse(predictions=[_predict(item) for item in batch.applications])
