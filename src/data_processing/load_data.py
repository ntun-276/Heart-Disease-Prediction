from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import pandas as pd

from src.utils.helpers import get_project_root

EXPECTED_COLUMNS = {
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
}


def load_heart_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    if data_path is None:
        data_path = get_project_root() / "data" / "heart.csv"

    data = cast(pd.DataFrame, pd.read_csv(data_path))
    missing_columns = EXPECTED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {sorted(missing_columns)}")

    return data


