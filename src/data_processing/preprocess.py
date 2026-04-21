from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ONE_HOT_COLUMNS = ["cp", "restecg", "thal"]
INT_COLUMNS = ["sex", "fbs", "exang", "slope", "ca", "target"]


@dataclass
class ProcessedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    lambdas: dict[str, float]


def encode_features(data: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(data, columns=ONE_HOT_COLUMNS, drop_first=True)
    for column in INT_COLUMNS:
        encoded[column] = encoded[column].astype(int)
    return encoded


def split_data(
    encoded_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = encoded_data.drop(columns=["target"])
    y = encoded_data["target"]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def apply_boxcox_transform(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    continuous_features: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if continuous_features is None:
        continuous_features = CONTINUOUS_FEATURES

    transformed_train = X_train.copy()
    transformed_test = X_test.copy()
    lambdas: dict[str, float] = {}

    # Box-Cox requires strictly positive values, so shift columns when needed.
    for column in continuous_features:
        train_column = transformed_train[column].astype(float)
        test_column = transformed_test[column].astype(float)

        min_value = min(train_column.min(), test_column.min())
        shift = abs(min_value) + 1e-3 if min_value <= 0 else 0.0

        shifted_train = train_column + shift
        shifted_test = test_column + shift

        transformed_train[column], lambdas[column] = boxcox(shifted_train)
        transformed_test[column] = boxcox(shifted_test, lmbda=lambdas[column])

    return transformed_train, transformed_test, lambdas


def preprocess_data(data: pd.DataFrame) -> ProcessedData:
    encoded = encode_features(data)
    X_train, X_test, y_train, y_test = split_data(encoded)
    X_train, X_test, lambdas = apply_boxcox_transform(X_train, X_test)
    return ProcessedData(X_train, X_test, y_train, y_test, lambdas)