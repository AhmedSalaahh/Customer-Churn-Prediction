"""
scripts/train.py
-----------------
CLI entry point that runs the full training pipeline:
  1. Load & clean data
  2. Feature engineering
  3. Preprocess + SMOTE
  4. Train all models with MLflow tracking
  5. Select best model & persist artefacts
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import yaml

# Make sure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import clean_data, load_raw_data, split_data
from src.evaluation.metrics import (
    evaluate_model,
    intervention_simulation,
    plot_pr_curves,
)
from src.features.engineering import build_features
from src.features.pipeline import (
    build_preprocessor,
    fit_transform_with_smote,
    get_feature_names,
)
from src.models.registry import save_model_artifact, select_best_model
from src.models.training import train_all_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict, data_path: str | None = None) -> None:
    # ── 0. MLflow setup ───────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["project"]["mlflow_tracking_uri"])
    logger.info("MLflow tracking URI: %s", cfg["project"]["mlflow_tracking_uri"])

    # ── 1. Load & clean ───────────────────────────────────────────────────
    raw_path = data_path or cfg["data"]["raw_path"]
    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw, cfg["data"])

    # ── 2. Feature engineering ────────────────────────────────────────────
    df_features = build_features(df_clean, cfg["features"])

    # ── 3. Split ──────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(
        df_features,
        target_col=cfg["data"]["target_col"],
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    # ── 4. Preprocess + SMOTE ─────────────────────────────────────────────
    numeric_features = [
        c for c in cfg["features"]["numeric"] if c in X_train.columns
    ]
    categorical_features = [
        c for c in cfg["features"]["categorical"] if c in X_train.columns
    ]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_res, y_train_res = fit_transform_with_smote(
        preprocessor, X_train, y_train,
        random_state=cfg["data"]["random_state"],
    )
    X_test_proc = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    # ── 5. Train all models ───────────────────────────────────────────────
    results = train_all_models(
        X_train=X_train_res,
        y_train=y_train_res,
        X_test=X_test_proc,
        y_test=y_test,
        cfg=cfg,
    )

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    fitted_models = {name: payload["model"] for name, payload in results.items()}
    for name, model in fitted_models.items():
        evaluate_model(model, X_test_proc, y_test, model_name=name)

    plot_pr_curves(fitted_models, X_test_proc, y_test, save_path="outputs/pr_curves.png")

    # ── 7. Select & persist best model ────────────────────────────────────
    best_name, best_model, best_roc = select_best_model(
        results, X_test_proc, y_test
    )
    logger.info("Champion: %s | ROC-AUC=%.4f", best_name, best_roc)

    save_model_artifact(
        model=best_model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        path="outputs/best_model.pkl",
    )

    # ── 8. Business simulation ────────────────────────────────────────────
    biz = cfg["business"]
    intervention_simulation(
        best_model,
        X_test_proc,
        y_test,
        threshold=0.65,
        retention_effectiveness=biz["retention_effectiveness"],
        retention_cost=biz["retention_cost"],
        revenue_per_customer=biz["revenue_per_customer"],
    )

    logger.info("Pipeline complete ✓")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Override data path from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, data_path=args.data)
