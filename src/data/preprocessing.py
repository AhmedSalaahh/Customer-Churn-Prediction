"""
data/preprocessing.py
---------------------
Handles raw data loading, cleaning, and train/test splitting.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and apply minimal type fixes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def clean_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply column drops, type coercions, and binary target encoding.

    Parameters
    ----------
    df  : raw DataFrame
    cfg : 'data' section of config.yaml
    """
    df = df.copy()

    # Drop non-predictive columns
    drop_cols = [c for c in cfg.get("drop_cols", []) if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        logger.debug("Dropped columns: %s", drop_cols)

    # Coerce TotalCharges → numeric (blank strings become NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Binary-encode the target
    target = cfg["target_col"]
    df[target] = df[target].map({"Yes": 1, "No": 0})

    na_count = df.isna().sum().sum()
    logger.info("NaN count after cleaning: %d", na_count)

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Split → train: %d, test: %d (pos rate: %.2f%%)",
        len(X_train),
        len(X_test),
        100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test
