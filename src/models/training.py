"""
models/training.py
------------------
Train, calibrate, and MLflow-log all three models.
"""

import logging

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────

def build_logistic_regression(cfg: dict) -> LogisticRegression:
    return LogisticRegression(**cfg)


def build_random_forest(cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**cfg)


def build_xgboost(cfg: dict) -> XGBClassifier:
    return XGBClassifier(**cfg)


# ──────────────────────────────────────────────
# Training + calibration
# ──────────────────────────────────────────────

def train_and_calibrate(
    model,
    X_train_processed: np.ndarray,
    y_train,
    calibration_method: str = "isotonic",
    calibration_cv: int = 3,
):
    """
    Fit the base model then wrap with probability calibration.

    Returns the calibrated estimator.
    """
    model.fit(X_train_processed, y_train)

    calibrated = CalibratedClassifierCV(
        model, method=calibration_method, cv=calibration_cv
    )
    calibrated.fit(X_train_processed, y_train)
    return calibrated


# ──────────────────────────────────────────────
# MLflow experiment runner
# ──────────────────────────────────────────────

def run_experiment(
    experiment_name: str,
    model_name: str,
    model,
    X_train: np.ndarray,
    y_train,
    X_test: np.ndarray,
    y_test,
    calibration_cfg: dict,
    extra_params: dict | None = None,
) -> tuple:
    """
    Train + calibrate inside a single MLflow run, log params/metrics/model.

    Returns (calibrated_model, run_id)
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name) as run:
        # ── log hyper-params ──────────────────
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("calibration_method", calibration_cfg["method"])
        mlflow.log_param("calibration_cv", calibration_cfg["cv"])
        if extra_params:
            mlflow.log_params(extra_params)

        # ── train ─────────────────────────────
        calibrated = train_and_calibrate(
            model,
            X_train,
            y_train,
            calibration_method=calibration_cfg["method"],
            calibration_cv=calibration_cfg["cv"],
        )

        # ── evaluate ──────────────────────────
        y_proba = calibrated.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("pr_auc", pr_auc)
        logger.info("%s → ROC-AUC: %.4f | PR-AUC: %.4f", model_name, roc, pr_auc)

        # ── log artefact ──────────────────────
        mlflow.sklearn.log_model(calibrated, artifact_path="model")

        run_id = run.info.run_id

    return calibrated, run_id


# ──────────────────────────────────────────────
# High-level orchestrator
# ──────────────────────────────────────────────

def train_all_models(
    X_train: np.ndarray,
    y_train,
    X_test: np.ndarray,
    y_test,
    cfg: dict,
) -> dict:
    """
    Build, train, calibrate, and MLflow-log all three models.

    Parameters
    ----------
    cfg : full config dict (uses 'models', 'calibration', 'project' keys)

    Returns
    -------
    dict  {model_name: {"model": <calibrated>, "run_id": str}}
    """
    experiment_name = cfg["project"]["experiment_name"]
    calibration_cfg = cfg["calibration"]
    results = {}

    # ── Logistic Regression ───────────────────
    lr = build_logistic_regression(cfg["models"]["logistic_regression"])
    calibrated_lr, run_id_lr = run_experiment(
        experiment_name=experiment_name,
        model_name="Logistic Regression",
        model=lr,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        calibration_cfg=calibration_cfg,
        extra_params=cfg["models"]["logistic_regression"],
    )
    results["logistic_regression"] = {"model": calibrated_lr, "run_id": run_id_lr}

    # ── Random Forest ─────────────────────────
    rf = build_random_forest(cfg["models"]["random_forest"])
    calibrated_rf, run_id_rf = run_experiment(
        experiment_name=experiment_name,
        model_name="Random Forest",
        model=rf,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        calibration_cfg=calibration_cfg,
        extra_params=cfg["models"]["random_forest"],
    )
    results["random_forest"] = {"model": calibrated_rf, "run_id": run_id_rf}

    # ── XGBoost ───────────────────────────────
    xgb = build_xgboost(cfg["models"]["xgboost"])
    calibrated_xgb, run_id_xgb = run_experiment(
        experiment_name=experiment_name,
        model_name="XGBoost",
        model=xgb,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        calibration_cfg=calibration_cfg,
        extra_params=cfg["models"]["xgboost"],
    )
    results["xgboost"] = {"model": calibrated_xgb, "run_id": run_id_xgb}

    return results
