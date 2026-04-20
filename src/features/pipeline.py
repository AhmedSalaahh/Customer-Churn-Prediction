"""
features/pipeline.py
---------------------
Builds the sklearn ColumnTransformer + SMOTE preprocessing pipeline.
"""

import logging
import numpy as np
from typing import Tuple

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Create a ColumnTransformer that:
    - Imputes + scales numeric features
    - Imputes + one-hot-encodes categorical features
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def fit_transform_with_smote(
    preprocessor: ColumnTransformer,
    X_train,
    y_train,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit the preprocessor on X_train, apply SMOTE to balance classes.

    Returns
    -------
    X_resampled, y_resampled : balanced training arrays
    """
    X_train_processed = preprocessor.fit_transform(X_train)
    logger.info("Preprocessor fitted. Shape before SMOTE: %s", X_train_processed.shape)

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)
    logger.info("SMOTE applied. Shape after SMOTE: %s", X_resampled.shape)

    return X_resampled, y_resampled


def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: list,
    categorical_features: list,
) -> list:
    """Return the full ordered list of feature names after OHE."""
    ohe = preprocessor.named_transformers_["cat"]["encoder"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    return list(numeric_features) + cat_names
