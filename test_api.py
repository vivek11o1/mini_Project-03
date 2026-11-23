"""Quick smoke test: python test_api.py"""

import warnings

warnings.filterwarnings("ignore", message=".*httpx.*deprecated.*")
warnings.filterwarnings("ignore", message=".*httpx2.*")

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

client = TestClient(app)

APPROVED_EXAMPLE = {
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
RISKY_EXAMPLE = {
    "no_of_dependents": 0,
    "education": "Not Graduate",
    "self_employed": "Yes",
    "income_annum": 4100000,
    "loan_amount": 12200000,
    "loan_term": 8,
    "cibil_score": 417,
    "residential_assets_value": 2700000,
    "commercial_assets_value": 2200000,
    "luxury_assets_value": 8800000,
    "bank_asset_value": 3300000,
}

health = client.get("/health")
assert health.status_code == 200, health.text
assert health.json()["status"] == "ok"
print("health:", health.json())

first = client.post("/predict", json=APPROVED_EXAMPLE)
assert first.status_code == 200, first.text
assert first.json()["status"] in ("Approved", "Rejected")
assert 0.0 <= first.json()["confidence"] <= 1.0
print("example-1:", first.json())

second = client.post("/predict", json=RISKY_EXAMPLE)
assert second.status_code == 200, second.text
print("example-2:", second.json())

batch = client.post("/predict/batch", json={"applications": [APPROVED_EXAMPLE, RISKY_EXAMPLE]})
assert batch.status_code == 200, batch.text
assert len(batch.json()["predictions"]) == 2
print("batch:", batch.json())

invalid = client.post("/predict", json={**APPROVED_EXAMPLE, "education": "PhD"})
assert invalid.status_code == 422
print("invalid education: correctly rejected with 422")

assert client.get("/docs").status_code == 200
print("Swagger UI: /docs OK")
print("ALL TESTS PASSED")
