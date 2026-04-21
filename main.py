from __future__ import annotations

from src.evaluation.evaluate import compare_models
from src.feature_engineering.build_features import build_feature_set
from src.models.train import train_models
from src.utils.helpers import ensure_dir, get_project_root, save_json, save_model


def run_pipeline() -> None:
    feature_set = build_feature_set()
    trained_models = train_models(feature_set.X_train, feature_set.y_train)

    model_pairs = [(trained.name, trained.estimator) for trained in trained_models]
    evaluation_table = compare_models(model_pairs, feature_set.X_test, feature_set.y_test)

    root = get_project_root()
    model_dir = ensure_dir(root / "models")
    report_dir = ensure_dir(root / "reports")

    for trained in trained_models:
        save_model(trained.estimator, model_dir / f"{trained.name.lower()}_model.pkl")

    best_model_name = evaluation_table.index[0]
    best_params = next(item.best_params for item in trained_models if item.name == best_model_name)

    evaluation_table.to_csv(report_dir / "model_results.csv", index=True)
    save_json({"best_model": best_model_name, "best_params": best_params}, report_dir / "best_model.json")

    print("Training finished. Evaluation table:")
    print(evaluation_table)
    print(f"\nSaved models to: {model_dir}")
    print(f"Saved reports to: {report_dir}")


if __name__ == "__main__":
    run_pipeline()


