"""
tests/test_pipeline.py
-----------------------
Tests for the sklearn ColumnTransformer + SMOTE pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.pipeline import (
    build_preprocessor,
    fit_transform_with_smote,
    get_feature_names,
)

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
                    "avg_monthly_spend", "contract_risk", "num_services", "high_value_customer"]

CATEGORICAL_FEATURES = ["gender", "Contract", "PaymentMethod"]


@pytest.fixture()
def small_x():
    """40-row synthetic frame – enough for SMOTE (needs ≥ k_neighbors=5 minority samples)."""
    np.random.seed(0)
    n = 40
    return pd.DataFrame({
        "SeniorCitizen": np.random.randint(0, 2, n),
        "tenure": np.random.randint(1, 72, n),
        "MonthlyCharges": np.random.uniform(20, 120, n),
        "TotalCharges": np.random.uniform(20, 8000, n),
        "avg_monthly_spend": np.random.uniform(10, 120, n),
        "contract_risk": np.random.choice([0, 1, 2], n),
        "num_services": np.random.randint(0, 9, n),
        "high_value_customer": np.random.randint(0, 2, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)"], n
        ),
    })


@pytest.fixture()
def small_y():
    """Imbalanced binary labels (25 zeros, 15 ones)."""
    return pd.Series([0] * 25 + [1] * 15, name="Churn")


# ──────────────────────────────────────────────
# build_preprocessor
# ──────────────────────────────────────────────

class TestBuildPreprocessor:
    def test_returns_column_transformer(self, small_x, small_y):
        from sklearn.compose import ColumnTransformer
        prep = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        assert isinstance(prep, ColumnTransformer)

    def test_fit_transform_produces_array(self, small_x, small_y):
        prep = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        out = prep.fit_transform(small_x)
        assert isinstance(out, np.ndarray)

    def test_output_rows_match_input(self, small_x, small_y):
        prep = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        out = prep.fit_transform(small_x)
        assert out.shape[0] == len(small_x)

    def test_output_has_more_cols_than_input(self, small_x, small_y):
        """OHE should expand the categorical columns."""
        prep = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        out = prep.fit_transform(small_x)
        assert out.shape[1] > small_x.shape[1]


# ──────────────────────────────────────────────
# fit_transform_with_smote
# ──────────────────────────────────────────────

class TestFitTransformWithSMOTE:
    def test_balances_classes(self, small_x, small_y):
        prep = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        X_res, y_res = fit_transform_with_smote(prep, small_x, small_y, random_state=42)
        counts = pd.Series(y_res).value_counts()
        assert counts[0] == counts[1], "Classes should be balanced after SMOTE"

    def test_resampled_larger_than_original(self, small_x, small_y):
        prep = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        X_res, y_res = fit_transform_with_smote(prep, small_x, small_y, random_state=42)
        assert len(X_res) >= len(small_x), "Resampled data should have at least as many rows as original"

    def test_no_nan_in_output(self, small_x, small_y):
        prep = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        X_res, _ = fit_transform_with_smote(prep, small_x, small_y, random_state=42)
        assert not np.isnan(X_res).any()


# ──────────────────────────────────────────────
# get_feature_names
# ──────────────────────────────────────────────

class TestGetFeatureNames:
    def test_length_matches_array_cols(self, small_x, small_y):
        prep = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        X_proc = prep.fit_transform(small_x)
        names = get_feature_names(prep, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        assert len(names) == X_proc.shape[1]

    def test_numeric_features_first(self, small_x, small_y):
        prep = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        prep.fit_transform(small_x)
        names = get_feature_names(prep, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        for i, feat in enumerate(NUMERIC_FEATURES):
            assert names[i] == feat
