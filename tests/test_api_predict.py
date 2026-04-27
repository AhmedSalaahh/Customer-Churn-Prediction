"""
tests/test_api_predict.py
--------------------------
Tests for /predict and /predict/batch endpoints.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _make_dynamic_bundle(proba: float = 0.72) -> dict:
    """
    Bundle whose predict_proba dynamically matches the number of rows
    in the transformed array — no fixed n hardcoded.
    """
    fake_model = MagicMock()

    def dynamic_predict_proba(X):
        n = X.shape[0]
        return np.array([[1 - proba, proba]] * n)

    fake_model.predict_proba.side_effect = dynamic_predict_proba

    fake_preprocessor = MagicMock()

    def dynamic_transform(df):
        n = len(df)
        return np.zeros((n, 42))

    fake_preprocessor.transform.side_effect = dynamic_transform

    return {
        "model": fake_model,
        "preprocessor": fake_preprocessor,
        "feature_names": [f"feat_{i}" for i in range(42)],
    }


@pytest.fixture(scope="module")
def client():
    from src.api.app import create_app
    with patch("src.api.inference.load_model_artifact", return_value=_make_dynamic_bundle()):
        app = create_app(
            config_path="configs/config.yaml",
            model_path="fake/path/model.pkl",
            threshold=0.5,
        )
        with TestClient(app) as c:
            yield c


class TestPredictSingle:
    def test_returns_200(self, client):
        assert client.post("/api/v1/predict", json=VALID_CUSTOMER).status_code == 200

    def test_response_shape(self, client):
        data = client.post("/api/v1/predict", json=VALID_CUSTOMER).json()
        assert data["status"] == "ok"
        for key in ("churn_probability", "churn_predicted", "risk_tier", "expected_revenue_loss"):
            assert key in data["prediction"]

    def test_probability_in_range(self, client):
        prob = client.post("/api/v1/predict", json=VALID_CUSTOMER).json()["prediction"]["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_tier_valid(self, client):
        tier = client.post("/api/v1/predict", json=VALID_CUSTOMER).json()["prediction"]["risk_tier"]
        assert tier in {"Low", "Medium", "High"}

    def test_high_probability_sets_predicted_true(self, client):
        pred = client.post("/api/v1/predict", json=VALID_CUSTOMER).json()["prediction"]
        assert pred["churn_predicted"] is True  # mock returns 0.72 > 0.5

    def test_expected_revenue_loss_non_negative(self, client):
        loss = client.post("/api/v1/predict", json=VALID_CUSTOMER).json()["prediction"]["expected_revenue_loss"]
        assert loss >= 0


class TestPredictSingleValidation:
    def test_missing_field_returns_422(self, client):
        bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
        assert client.post("/api/v1/predict", json=bad).status_code == 422

    def test_invalid_gender_returns_422(self, client):
        assert client.post("/api/v1/predict", json={**VALID_CUSTOMER, "gender": "Unknown"}).status_code == 422

    def test_invalid_contract_returns_422(self, client):
        assert client.post("/api/v1/predict", json={**VALID_CUSTOMER, "Contract": "Weekly"}).status_code == 422

    def test_negative_monthly_charges_returns_422(self, client):
        assert client.post("/api/v1/predict", json={**VALID_CUSTOMER, "MonthlyCharges": -10.0}).status_code == 422

    def test_senior_citizen_out_of_range_returns_422(self, client):
        assert client.post("/api/v1/predict", json={**VALID_CUSTOMER, "SeniorCitizen": 5}).status_code == 422

    def test_empty_body_returns_422(self, client):
        assert client.post("/api/v1/predict", json={}).status_code == 422


class TestPredictBatch:
    def test_returns_200(self, client):
        r = client.post("/api/v1/predict/batch", json={"customers": [VALID_CUSTOMER, VALID_CUSTOMER]})
        assert r.status_code == 200

    def test_total_matches_input(self, client):
        data = client.post("/api/v1/predict/batch", json={"customers": [VALID_CUSTOMER] * 3}).json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3

    def test_each_prediction_has_required_fields(self, client):
        data = client.post("/api/v1/predict/batch", json={"customers": [VALID_CUSTOMER] * 2}).json()
        for pred in data["predictions"]:
            for key in ("churn_probability", "churn_predicted", "risk_tier"):
                assert key in pred

    def test_empty_customers_returns_422(self, client):
        assert client.post("/api/v1/predict/batch", json={"customers": []}).status_code == 422

    def test_single_customer_in_batch(self, client):
        data = client.post("/api/v1/predict/batch", json={"customers": [VALID_CUSTOMER]}).json()
        assert data["total"] == 1
