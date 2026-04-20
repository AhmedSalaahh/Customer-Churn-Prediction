"""
models/registry.py
------------------
Save, load, and select the best model from MLflow runs.
"""

import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def select_best_model(results: dict, X_test: np.ndarray, y_test) -> tuple:
    """
    Pick the model with the highest ROC-AUC on the test set.

    Parameters
    ----------
    results : output of train_all_models()

    Returns
    -------
    (best_name, best_model, best_roc)
    """
    best_name, best_model, best_roc = None, None, -1.0

    for name, payload in results.items():
        model = payload["model"]
        roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        logger.info("%s test ROC-AUC: %.4f", name, roc)
        if roc > best_roc:
            best_name, best_model, best_roc = name, model, roc

    logger.info("Best model: %s (ROC-AUC %.4f)", best_name, best_roc)
    return best_name, best_model, best_roc


def save_model_artifact(model, preprocessor, feature_names: list, path: str) -> None:
    """
    Persist the calibrated model + preprocessor + feature names together
    as a single pickle bundle so inference needs no extra config.
    """
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Model bundle saved to %s", path)


def load_model_artifact(path: str) -> dict:
    """Load the bundle saved by save_model_artifact."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info("Model bundle loaded from %s", path)
    return bundle


def register_best_model_in_mlflow(
    run_id: str,
    model_name: str,
    tracking_uri: str = "mlruns",
) -> None:
    """
    Register a run's model in the MLflow Model Registry.
    Useful once you add a tracking server.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info("Registered model '%s' from run %s", model_name, run_id)
