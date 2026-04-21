from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data_processing.load_data import load_heart_data
from src.data_processing.preprocess import ProcessedData, preprocess_data


@dataclass
class FeatureSet:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    lambdas: dict[str, float]


def build_feature_set() -> FeatureSet:
    raw_data = load_heart_data()
    processed: ProcessedData = preprocess_data(raw_data)

    return FeatureSet(
        X_train=processed.X_train,
        X_test=processed.X_test,
        y_train=processed.y_train,
        y_test=processed.y_test,
        feature_names=processed.X_train.columns.tolist(),
        lambdas=processed.lambdas,
    )

