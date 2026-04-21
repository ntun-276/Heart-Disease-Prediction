from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class TrainedModel:
    name: str
    estimator: Any
    best_params: dict[str, Any]


def tune_clf_hyperparameters(
    clf: Any,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "recall",
    n_splits: int = 3,
) -> tuple[Any, dict[str, Any]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    clf_grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    clf_grid.fit(X_train, y_train)
    return clf_grid.best_estimator_, clf_grid.best_params_


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> list[TrainedModel]:
    model_configs = [
        (
            "DT",
            DecisionTreeClassifier(random_state=0),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [2, 3, 4],
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [1, 2],
            },
        ),
        (
            "RF",
            RandomForestClassifier(random_state=0),
            {
                "n_estimators": [30, 70, 120],
                "criterion": ["gini", "entropy"],
                "max_depth": [2, 3, 4],
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [1, 2],
                "bootstrap": [True, False],
            },
        ),
        (
            "KNN",
            Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier()),
            ]),
            {
                "knn__n_neighbors": list(range(1, 12)),
                "knn__weights": ["uniform", "distance"],
                "knn__p": [1, 2],
            },
        ),
        (
            "SVM",
            Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(probability=True, random_state=0)),
            ]),
            {
                "svm__C": [0.01, 0.1, 1, 10],
                "svm__kernel": ["linear", "rbf"],
                "svm__gamma": ["scale", "auto", 0.1],
            },
        ),
    ]

    trained_models: list[TrainedModel] = []
    for model_name, base_model, param_grid in model_configs:
        best_estimator, best_params = tune_clf_hyperparameters(
            clf=base_model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
        )
        trained_models.append(TrainedModel(model_name, best_estimator, best_params))

    return trained_models