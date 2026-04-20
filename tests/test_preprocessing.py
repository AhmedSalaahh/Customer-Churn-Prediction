"""
tests/test_preprocessing.py
----------------------------
Unit tests for src/data/preprocessing.py
"""

import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import clean_data, split_data


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture()
def raw_df():
    """Minimal synthetic dataset mirroring the Telco CSV schema."""
    return pd.DataFrame({
        "customerID": ["001", "002", "003", "004", "005", "006"],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0, 1, 0],
        "Partner": ["Yes", "No", "No", "Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes", "No", "Yes", "No"],
        "tenure": [1, 34, 2, 45, 2, 8],
        "PhoneService": ["No", "Yes", "Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "No", "No phone service", "No", "Yes"],
        "InternetService": ["DSL", "DSL", "DSL", "DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes", "Yes", "Yes", "No", "Yes"],
        "OnlineBackup": ["Yes", "No", "Yes", "No", "No", "No"],
        "DeviceProtection": ["No", "Yes", "No", "Yes", "No", "No"],
        "TechSupport": ["No", "No", "No", "Yes", "No", "No"],
        "StreamingTV": ["No", "No", "No", "No", "No", "No"],
        "StreamingMovies": ["No", "No", "No", "No", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "One year",
                     "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Mailed check",
                          "Bank transfer (automatic)", "Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 35.00],
        "TotalCharges": ["29.85", "1889.50", "108.15", "1840.75", "151.65", "280.00"],
        "Churn": ["No", "No", "Yes", "No", "Yes", "No"],
    })


@pytest.fixture()
def data_cfg():
    return {
        "drop_cols": ["customerID"],
        "target_col": "Churn",
        "test_size": 0.33,
        "random_state": 42,
    }


# ──────────────────────────────────────────────
# clean_data tests
# ──────────────────────────────────────────────

class TestCleanData:
    def test_drops_customer_id(self, raw_df, data_cfg):
        df = clean_data(raw_df, data_cfg)
        assert "customerID" not in df.columns

    def test_total_charges_is_numeric(self, raw_df, data_cfg):
        df = clean_data(raw_df, data_cfg)
        assert pd.api.types.is_float_dtype(df["TotalCharges"])

    def test_churn_is_binary_int(self, raw_df, data_cfg):
        df = clean_data(raw_df, data_cfg)
        assert set(df["Churn"].dropna().unique()).issubset({0, 1})

    def test_no_customerid_nan_introduced(self, raw_df, data_cfg):
        df = clean_data(raw_df, data_cfg)
        # Only TotalCharges may have NaN; everything else should be intact
        non_tc_na = df.drop(columns=["TotalCharges"]).isna().sum().sum()
        assert non_tc_na == 0

    def test_row_count_unchanged(self, raw_df, data_cfg):
        df = clean_data(raw_df, data_cfg)
        assert len(df) == len(raw_df)


# ──────────────────────────────────────────────
# split_data tests
# ──────────────────────────────────────────────

class TestSplitData:
    def test_split_sizes(self, raw_df, data_cfg):
        df = clean_data(raw_df, data_cfg)
        X_train, X_test, y_train, y_test = split_data(
            df,
            target_col=data_cfg["target_col"],
            test_size=data_cfg["test_size"],
            random_state=data_cfg["random_state"],
        )
        total = len(X_train) + len(X_test)
        assert total == len(df)

    def test_target_not_in_X(self, raw_df, data_cfg):
        df = clean_data(raw_df, data_cfg)
        X_train, X_test, _, _ = split_data(df, target_col=data_cfg["target_col"])
        assert data_cfg["target_col"] not in X_train.columns
        assert data_cfg["target_col"] not in X_test.columns

    def test_y_values_are_binary(self, raw_df, data_cfg):
        df = clean_data(raw_df, data_cfg)
        _, _, y_train, y_test = split_data(df, target_col=data_cfg["target_col"])
        for y in (y_train, y_test):
            assert set(y.unique()).issubset({0, 1})
