from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], file_path: Path) -> None:
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_model(model: Any, file_path: Path) -> None:
    ensure_dir(file_path.parent)
    joblib.dump(model, file_path)


def load_model(file_path: Path) -> Any:
    return joblib.load(file_path)

