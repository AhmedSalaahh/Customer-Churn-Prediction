"""
tests/test_engineering.py
--------------------------
Unit tests for src/features/engineering.py
"""

import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineering import (
    add_avg_monthly_spend,
    add_contract_risk,
    add_high_value_flag,
    add_num_services,
    build_features,
    normalize_service_cols,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def base_df():
    return pd.DataFrame({
        "tenure": [1, 10, 24, 0],   # 0 tenure to test clamp
        "TotalCharges": [30.0, 500.0, 2400.0, 0.0],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "PhoneService": ["No", "Yes", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No internet service", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No internet service", "Yes"],
        "OnlineBackup": ["Yes", "No", "No internet service", "No"],
        "DeviceProtection": ["No", "No", "No internet service", "No"],
        "TechSupport": ["No", "Yes", "No internet service", "No"],
        "StreamingTV": ["No", "No", "No internet service", "No"],
        "StreamingMovies": ["No", "No", "No internet service", "No"],
        "MultipleLines": ["No phone service", "Yes", "No phone service", "No"],
    })


CONTRACT_RISK_MAP = {"Month-to-month": 2, "One year": 1, "Two year": 0}

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

FEATURES_CFG = {
    "service_cols": SERVICE_COLS,
    "contract_risk_map": CONTRACT_RISK_MAP,
}


# ──────────────────────────────────────────────
# avg_monthly_spend
# ──────────────────────────────────────────────

class TestAvgMonthlySpend:
    def test_basic_calculation(self, base_df):
        df = add_avg_monthly_spend(base_df)
        assert "avg_monthly_spend" in df.columns
        # row 1: 500 / 10 = 50
        assert df.loc[1, "avg_monthly_spend"] == pytest.approx(50.0)

    def test_zero_tenure_clamped(self, base_df):
        df = add_avg_monthly_spend(base_df)
        # row 3: 0.0 / max(0,1) = 0.0 → no ZeroDivisionError
        assert not pd.isna(df.loc[3, "avg_monthly_spend"])

    def test_does_not_mutate_original(self, base_df):
        original_cols = list(base_df.columns)
        add_avg_monthly_spend(base_df)
        assert list(base_df.columns) == original_cols


# ──────────────────────────────────────────────
# contract_risk
# ──────────────────────────────────────────────

class TestContractRisk:
    def test_mapping_applied(self, base_df):
        df = add_contract_risk(base_df, CONTRACT_RISK_MAP)
        assert df.loc[0, "contract_risk"] == 2   # Month-to-month
        assert df.loc[1, "contract_risk"] == 1   # One year
        assert df.loc[2, "contract_risk"] == 0   # Two year

    def test_column_exists(self, base_df):
        df = add_contract_risk(base_df, CONTRACT_RISK_MAP)
        assert "contract_risk" in df.columns


# ──────────────────────────────────────────────
# normalize_service_cols
# ──────────────────────────────────────────────

class TestNormalizeServiceCols:
    def test_no_internet_service_replaced(self, base_df):
        df = normalize_service_cols(base_df, SERVICE_COLS)
        assert "No internet service" not in df["InternetService"].values

    def test_no_phone_service_replaced(self, base_df):
        df = normalize_service_cols(base_df, SERVICE_COLS)
        assert "No phone service" not in df["PhoneService"].values
        assert "No phone service" not in df["MultipleLines"].values

    def test_yes_values_preserved(self, base_df):
        df = normalize_service_cols(base_df, SERVICE_COLS)
        assert "Yes" in df["PhoneService"].values


# ──────────────────────────────────────────────
# num_services
# ──────────────────────────────────────────────

class TestNumServices:
    def test_counts_yes_values(self, base_df):
        df = normalize_service_cols(base_df, SERVICE_COLS)
        df = add_num_services(df, SERVICE_COLS)
        assert "num_services" in df.columns
        assert (df["num_services"] >= 0).all()

    def test_range(self, base_df):
        df = normalize_service_cols(base_df, SERVICE_COLS)
        df = add_num_services(df, SERVICE_COLS)
        assert df["num_services"].max() <= len(SERVICE_COLS)


# ──────────────────────────────────────────────
# high_value_flag
# ──────────────────────────────────────────────

class TestHighValueFlag:
    def test_column_is_binary(self, base_df):
        df = add_high_value_flag(base_df)
        assert set(df["high_value_customer"].unique()).issubset({0, 1})

    def test_column_exists(self, base_df):
        df = add_high_value_flag(base_df)
        assert "high_value_customer" in df.columns


# ──────────────────────────────────────────────
# build_features (orchestrator)
# ──────────────────────────────────────────────

class TestBuildFeatures:
    def test_all_new_columns_present(self, base_df):
        df = build_features(base_df, FEATURES_CFG)
        for col in ["avg_monthly_spend", "contract_risk", "num_services", "high_value_customer"]:
            assert col in df.columns, f"Missing: {col}"

    def test_row_count_unchanged(self, base_df):
        df = build_features(base_df, FEATURES_CFG)
        assert len(df) == len(base_df)
