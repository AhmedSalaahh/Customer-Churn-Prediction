"""
scripts/predict.py
------------------
Batch inference: load the saved model bundle and score a CSV of customers.

Usage:
    python scripts/predict.py \
        --model outputs/best_model.pkl \
        --input data/new_customers.csv \
        --output outputs/predictions.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.preprocessing import clean_data
from src.features.engineering import build_features
from src.models.registry import load_model_artifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_batch(model_path: str, input_path: str, output_path: str, config_path: str) -> None:
    cfg = load_config(config_path)

    # Load model bundle
    bundle = load_model_artifact(model_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    # Load and prepare input data
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows for scoring", len(df))

    # Apply same cleaning and feature engineering
    df_clean = clean_data(df, cfg["data"])
    df_feat = build_features(df_clean, cfg["features"])

    # Drop target if present
    target = cfg["data"]["target_col"]
    if target in df_feat.columns:
        df_feat = df_feat.drop(columns=[target])

    # Align columns to training schema
    X = preprocessor.transform(df_feat)

    # Score
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    # Output
    df_out = df.copy()
    df_out["churn_probability"] = proba
    df_out["churn_predicted"] = pred

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    logger.info("Predictions saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch churn prediction")
    parser.add_argument("--model", default="outputs/best_model.pkl")
    parser.add_argument("--input", required=True, help="CSV of customers to score")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    run_batch(args.model, args.input, args.output, args.config)
