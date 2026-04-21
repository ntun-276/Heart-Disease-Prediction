from __future__ import annotations

from typing import Any
from typing import cast

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    report = cast(
        dict[str, dict[str, float]],
        classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    )

    metrics = {
        "precision_0": report["0"]["precision"],
        "precision_1": report["1"]["precision"],
        "recall_0": report["0"]["recall"],
        "recall_1": report["1"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "f1_1": report["1"]["f1-score"],
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "accuracy": accuracy_score(y_test, y_pred),
    }

    return pd.DataFrame(metrics, index=[model_name]).round(2)


def compare_models(
    models: list[tuple[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    evaluations = [
        evaluate_model(model=estimator, X_test=X_test, y_test=y_test, model_name=name)
        for name, estimator in models
    ]
    results = pd.concat(evaluations)
    return results.sort_values(by="recall_1", ascending=False)


