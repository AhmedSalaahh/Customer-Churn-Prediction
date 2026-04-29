"""
tests/test_training_integration.py
------------------------------------
Integration test: run the full training pipeline on synthetic data
and assert that model artefacts and MLflow runs are produced correctly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import clean_data, split_data
from src.features.engineering import build_features
from src.features.pipeline import (
    build_preprocessor,
    fit_transform_with_smote,
    get_feature_names,
)
from src.models.registry import (
    load_model_artifact,
    save_model_artifact,
    select_best_model,
)
from src.models.training import train_all_models

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_synthetic_df(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Create a minimal Telco-like DataFrame with all required columns."""
    rng = np.random.default_rng(seed)
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], n)
    total_charges = rng.uniform(20, 8000, n)
    tenure = rng.integers(1, 72, n)

    return pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": contracts,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"], n
        ),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "TotalCharges": [str(v) for v in total_charges],  # raw CSV has strings
        "Churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })


@pytest.fixture(scope="module")
def mini_cfg(tmp_path_factory):
    """Config dict pointing at a temp MLflow tracking dir."""
    tmp = tmp_path_factory.mktemp("mlruns")
    return {
        "project": {
            "mlflow_tracking_uri": str(tmp),
            "experiment_name": "churn_test",
        },
        "data": {
            "drop_cols": ["customerID"],
            "target_col": "Churn",
            "test_size": 0.25,
            "random_state": 42,
        },
        "features": {
            "numeric": [
                "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
                "avg_monthly_spend", "contract_risk", "num_services", "high_value_customer",
            ],
            "categorical": [
                "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod",
            ],
            "service_cols": [
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
            ],
            "contract_risk_map": {
                "Month-to-month": 2,
                "One year": 1,
                "Two year": 0,
            },
        },
        "models": {
            "logistic_regression": {"max_iter": 500},
            "random_forest": {"n_estimators": 10, "max_depth": 3},
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "eval_metric": "logloss",
            },
        },
        "calibration": {"method": "isotonic", "cv": 3},
    }


# ──────────────────────────────────────────────
# Integration test
# ──────────────────────────────────────────────

class TestFullPipeline:
    @pytest.fixture(autouse=True, scope="class")
    def pipeline_output(self, mini_cfg, tmp_path_factory):
        """Run the full pipeline once; cache results for all tests in this class."""
        import mlflow
        mlflow.set_tracking_uri(mini_cfg["project"]["mlflow_tracking_uri"])

        raw = _make_synthetic_df()
        clean = clean_data(raw, mini_cfg["data"])
        feat = build_features(clean, mini_cfg["features"])

        X_train, X_test, y_train, y_test = split_data(
            feat, target_col="Churn",
            test_size=mini_cfg["data"]["test_size"],
            random_state=mini_cfg["data"]["random_state"],
        )

        num_feats = [c for c in mini_cfg["features"]["numeric"] if c in X_train.columns]
        cat_feats = [c for c in mini_cfg["features"]["categorical"] if c in X_train.columns]

        prep = build_preprocessor(num_feats, cat_feats)
        X_tr_res, y_tr_res = fit_transform_with_smote(prep, X_train, y_train, random_state=42)
        X_te_proc = prep.transform(X_test)
        feature_names = get_feature_names(prep, num_feats, cat_feats)

        results = train_all_models(X_tr_res, y_tr_res, X_te_proc, y_test, mini_cfg)

        best_name, best_model, best_roc = select_best_model(results, X_te_proc, y_test)

        tmp = tmp_path_factory.mktemp("artifacts")
        model_path = str(tmp / "best_model.pkl")
        save_model_artifact(best_model, prep, feature_names, model_path)

        # Store on class for individual tests
        TestFullPipeline._results = results
        TestFullPipeline._best_name = best_name
        TestFullPipeline._best_model = best_model
        TestFullPipeline._best_roc = best_roc
        TestFullPipeline._model_path = model_path
        TestFullPipeline._X_test = X_te_proc
        TestFullPipeline._y_test = y_test
        TestFullPipeline._feature_names = feature_names

    def test_three_models_trained(self):
        assert len(self._results) == 3

    def test_all_models_have_run_ids(self):
        for name, payload in self._results.items():
            assert "run_id" in payload, f"No run_id for {name}"
            assert payload["run_id"]  # non-empty string

    def test_best_roc_is_reasonable(self):
        """ROC-AUC should be at least 0.5 (better than random) on synthetic data."""
        assert self._best_roc >= 0.5

    def test_model_bundle_saves_and_loads(self):
        bundle = load_model_artifact(self._model_path)
        assert "model" in bundle
        assert "preprocessor" in bundle
        assert "feature_names" in bundle

    def test_loaded_model_can_predict(self):
        bundle = load_model_artifact(self._model_path)
        model = bundle["model"]
        proba = model.predict_proba(self._X_test)[:, 1]
        assert len(proba) == len(self._y_test)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_feature_names_length_matches_array(self):
        assert len(self._feature_names) == self._X_test.shape[1]
