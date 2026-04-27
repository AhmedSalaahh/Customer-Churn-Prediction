"""
tests/test_api_health.py
-------------------------
Tests for /health and /model/info endpoints.
Strategy: patch load_model_artifact so no file I/O occurs.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fake_bundle(n_features: int = 42, proba: float = 0.7) -> dict:
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[1 - proba, proba]])
    fake_model.predict.return_value = np.array([int(proba >= 0.5)])
    fake_preprocessor = MagicMock()
    fake_preprocessor.transform.return_value = np.zeros((1, n_features))
    return {
        "model": fake_model,
        "preprocessor": fake_preprocessor,
        "feature_names": [f"feat_{i}" for i in range(n_features)],
    }


@pytest.fixture(scope="module")
def client():
    from src.api.app import create_app
    with patch("src.api.inference.load_model_artifact", return_value=_fake_bundle()):
        app = create_app(
            config_path="configs/config.yaml",
            model_path="fake/path/model.pkl",
            threshold=0.5,
        )
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/api/v1/health").status_code == 200

    def test_model_loaded_true(self, client):
        assert client.get("/api/v1/health").json()["model_loaded"] is True

    def test_status_ok(self, client):
        assert client.get("/api/v1/health").json()["status"] == "ok"

    def test_model_version_present(self, client):
        data = client.get("/api/v1/health").json()
        assert data["model_version"]


class TestModelInfo:
    def test_returns_200(self, client):
        assert client.get("/api/v1/model/info").status_code == 200

    def test_feature_count(self, client):
        assert client.get("/api/v1/model/info").json()["feature_count"] == 42

    def test_feature_names_list(self, client):
        names = client.get("/api/v1/model/info").json()["feature_names"]
        assert isinstance(names, list) and len(names) == 42

    def test_threshold_present(self, client):
        assert client.get("/api/v1/model/info").json()["threshold"] == 0.5

    def test_risk_tiers_present(self, client):
        tiers = client.get("/api/v1/model/info").json()["risk_tiers"]
        assert set(tiers.keys()) == {"Low", "Medium", "High"}


class TestRoot:
    def test_root_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_has_message(self, client):
        assert "message" in client.get("/").json()
