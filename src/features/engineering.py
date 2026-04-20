"""
features/engineering.py
------------------------
Domain-specific feature creation from the raw cleaned DataFrame.
"""

import logging
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges / tenure (clamp tenure ≥ 1 to avoid div-by-zero)."""
    df = df.copy()
    safe_tenure = df["tenure"].clip(lower=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / safe_tenure
    return df


def add_contract_risk(df: pd.DataFrame, risk_map: dict) -> pd.DataFrame:
    """Ordinal risk score derived from contract type."""
    df = df.copy()
    df["contract_risk"] = df["Contract"].map(risk_map)
    return df


def normalize_service_cols(df: pd.DataFrame, service_cols: List[str]) -> pd.DataFrame:
    """
    Replace 'No internet service' / 'No phone service' with 'No'
    so all service flags are simple Yes/No.
    """
    df = df.copy()
    for col in service_cols:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No internet service": "No", "No phone service": "No"}
            )
    return df


def add_num_services(df: pd.DataFrame, service_cols: List[str]) -> pd.DataFrame:
    """Count of active services per customer."""
    df = df.copy()
    existing = [c for c in service_cols if c in df.columns]
    df["num_services"] = (df[existing] == "Yes").sum(axis=1)
    return df


def add_high_value_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: TotalCharges > median AND tenure > median."""
    df = df.copy()
    df["high_value_customer"] = (
        (df["TotalCharges"] > df["TotalCharges"].median())
        & (df["tenure"] > df["tenure"].median())
    ).astype(int)
    return df


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Orchestrates all feature engineering steps.

    Parameters
    ----------
    df  : cleaned DataFrame (before preprocessing pipeline)
    cfg : 'features' section of config.yaml
    """
    service_cols = cfg["service_cols"]
    contract_risk_map = cfg["contract_risk_map"]

    df = normalize_service_cols(df, service_cols)
    df = add_avg_monthly_spend(df)
    df = add_contract_risk(df, contract_risk_map)
    df = add_num_services(df, service_cols)
    df = add_high_value_flag(df)

    logger.info(
        "Feature engineering complete. Shape: %s. Columns added: "
        "avg_monthly_spend, contract_risk, num_services, high_value_customer",
        df.shape,
    )
    return df
