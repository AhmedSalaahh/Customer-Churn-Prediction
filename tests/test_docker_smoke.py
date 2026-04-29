"""
tests/test_docker_smoke.py
---------------------------
Smoke tests that validate the container is working correctly.

These are NOT run in the normal pytest suite — they require a running
Docker container. Run them separately after `docker compose up`:

    pytest tests/test_docker_smoke.py \
        --base-url http://localhost:8000 -v

They test the real HTTP surface of the container (no mocks).
"""

import os

import requests

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API      = f"{BASE_URL}/api/v1"

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(path: str) -> requests.Response:
    return requests.get(f"{API}{path}", timeout=10)


def _post(path: str, body: dict) -> requests.Response:
    return requests.post(f"{API}{path}", json=body, timeout=10)


# ── Health ────────────────────────────────────────────────────────────────────

class TestContainerHealth:
    def test_root_reachable(self):
        r = requests.get(BASE_URL, timeout=5)
        assert r.status_code == 200

    def test_health_endpoint(self):
        r = _get("/health")
        assert r.status_code == 200

    def test_model_loaded(self):
        data = _get("/health").json()
        assert data["model_loaded"] is True, (
            "Model not loaded — did you mount outputs/best_model.pkl?"
        )

    def test_status_ok(self):
        assert _get("/health").json()["status"] == "ok"

    def test_docs_reachable(self):
        r = requests.get(f"{BASE_URL}/docs", timeout=5)
        assert r.status_code == 200


# ── Model info ────────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_info_endpoint(self):
        assert _get("/model/info").status_code == 200

    def test_feature_count_positive(self):
        data = _get("/model/info").json()
        assert data["feature_count"] > 0

    def test_risk_tiers_present(self):
        data = _get("/model/info").json()
        assert set(data["risk_tiers"].keys()) == {"Low", "Medium", "High"}


# ── Single prediction ─────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(self):
        assert _post("/predict", VALID_CUSTOMER).status_code == 200

    def test_probability_in_range(self):
        prob = _post("/predict", VALID_CUSTOMER).json()["prediction"]["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_tier_valid(self):
        tier = _post("/predict", VALID_CUSTOMER).json()["prediction"]["risk_tier"]
        assert tier in {"Low", "Medium", "High"}

    def test_invalid_input_returns_422(self):
        bad = {**VALID_CUSTOMER, "gender": "Robot"}
        assert _post("/predict", bad).status_code == 422


# ── Batch prediction ──────────────────────────────────────────────────────────

class TestBatchEndpoint:
    def test_batch_returns_200(self):
        payload = {"customers": [VALID_CUSTOMER, VALID_CUSTOMER]}
        assert _post("/predict/batch", payload).status_code == 200

    def test_batch_total_matches(self):
        payload = {"customers": [VALID_CUSTOMER] * 5}
        data = _post("/predict/batch", payload).json()
        assert data["total"] == 5
        assert len(data["predictions"]) == 5
