from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.helpers import load_model


def load_trained_model(model_path: Path) -> Any:
    return load_model(model_path)


def predict(model: Any, features: pd.DataFrame) -> pd.Series:
    predictions = model.predict(features)
    return pd.Series(predictions, index=features.index, name="prediction")


def predict_proba(model: Any, features: pd.DataFrame) -> pd.Series:
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The provided model does not support predict_proba.")

    probabilities = model.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=features.index, name="probability_1")

