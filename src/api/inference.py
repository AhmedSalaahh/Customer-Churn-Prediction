"""
src/api/inference.py
---------------------
Loads the model bundle once at startup and exposes predict functions.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models.registry import load_model_artifact

logger = logging.getLogger(__name__)

RISK_TIERS = {
    "Low":    (0.00, 0.35),
    "Medium": (0.35, 0.65),
    "High":   (0.65, 1.01),
}


def _assign_risk_tier(proba: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= proba < hi:
            return tier
    return "High"


# ── Singleton bundle ──────────────────────────────────────────────────────────

class ModelBundle:
    def __init__(self, model, preprocessor, feature_names: list):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        logger.info("ModelBundle ready — %d features", len(feature_names))

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


_bundle: Optional[ModelBundle] = None


def load_bundle(model_path: str) -> None:
    global _bundle
    raw = load_model_artifact(model_path)
    _bundle = ModelBundle(
        model=raw["model"],
        preprocessor=raw["preprocessor"],
        feature_names=raw["feature_names"],
    )
    logger.info("Bundle loaded from %s", model_path)


def get_bundle() -> ModelBundle:
    if _bundle is None:
        raise RuntimeError("Model bundle not loaded. Call load_bundle() during app startup.")
    return _bundle


# ── Feature engineering ───────────────────────────────────────────────────────

def _apply_feature_engineering(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    from src.features.engineering import build_features
    return build_features(df, cfg)


# ── Public predict functions ──────────────────────────────────────────────────

def predict_single(customer: dict, features_cfg: dict, threshold: float = 0.5) -> dict:
    bundle = get_bundle()

    df = pd.DataFrame([customer])
    df = _apply_feature_engineering(df, features_cfg)
    df = df.drop(columns=["Churn"], errors="ignore")

    X = bundle.preprocessor.transform(df)
    proba = float(bundle.model.predict_proba(X)[0, 1])
    tier = _assign_risk_tier(proba)
    tc = float(customer.get("TotalCharges", 0.0) or 0.0)

    return {
        "churn_probability": round(proba, 4),
        "churn_predicted": bool(proba >= threshold),
        "risk_tier": tier,
        "expected_revenue_loss": round(proba * tc, 2),
    }


def predict_batch(customers: list[dict], features_cfg: dict, threshold: float = 0.5) -> list[dict]:
    bundle = get_bundle()
    n = len(customers)

    df = pd.DataFrame(customers)
    df = _apply_feature_engineering(df, features_cfg)
    df = df.drop(columns=["Churn"], errors="ignore")

    X = bundle.preprocessor.transform(df)
    # predict_proba returns (n_samples, 2) — take column 1 (churn)
    probas = bundle.model.predict_proba(X)[:, 1]

    results = []
    for i in range(n):
        proba = float(probas[i])
        tc = float(customers[i].get("TotalCharges", 0.0) or 0.0)
        results.append({
            "churn_probability": round(proba, 4),
            "churn_predicted": bool(proba >= threshold),
            "risk_tier": _assign_risk_tier(proba),
            "expected_revenue_loss": round(proba * tc, 2),
        })

    return results
