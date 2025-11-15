import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from config import (
        MLFLOW_EXPERIMENT,
        MLFLOW_TRACKING_URI,
        MODEL_ARTIFACT_PATH,
        SPARK_OUTPUT_PATH,
    )
except ImportError:
    from scripts.config import (
        MLFLOW_EXPERIMENT,
        MLFLOW_TRACKING_URI,
        MODEL_ARTIFACT_PATH,
        SPARK_OUTPUT_PATH,
    )


def _to_feature_vector(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if isinstance(value, tuple):
        return np.asarray(list(value), dtype=float)
    raise TypeError(f"Unsupported feature format: {type(value)}")


def load_training_data(data_path: str):
    df = pd.read_parquet(data_path)
    if "features" not in df.columns or "label" not in df.columns:
        raise ValueError("Input parquet must contain 'features' and 'label' columns")

    x = np.vstack(df["features"].apply(_to_feature_vector).to_numpy())
    y = df["label"].astype(int).to_numpy()
    customer_ids = (
        df["customerID"].astype(str).to_numpy()
        if "customerID" in df.columns
        else np.arange(len(df)).astype(str)
    )
    return x, y, customer_ids


def train_and_log_model(data_path: str, artifacts_dir: str, random_state: int = 42) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    x, y, customer_ids = load_training_data(data_path)
    stratify = y if len(np.unique(y)) > 1 else None

    x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(
        x,
        y,
        customer_ids,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    with mlflow.start_run(run_name="random_forest_churn_model") as run:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
        )
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        if len(np.unique(y_test)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

        mlflow.log_params(
            {
                "n_estimators": 200,
                "max_depth": 12,
                "min_samples_leaf": 2,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path=MODEL_ARTIFACT_PATH)

        output_dir = Path(artifacts_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions_sample.csv"
        pd.DataFrame(
            {
                "customerID": ids_test,
                "actual_churn": y_test,
                "predicted_churn": y_pred,
                "churn_probability": y_prob,
            }
        ).head(200).to_csv(predictions_path, index=False)
        mlflow.log_artifact(str(predictions_path), artifact_path="predictions")

        print(f"Run ID: {run.info.run_id}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        return run.info.run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn model and log metrics/model in MLflow")
    parser.add_argument("--data-path", default=SPARK_OUTPUT_PATH)
    parser.add_argument("--artifacts-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_log_model(args.data_path, args.artifacts_dir)


if __name__ == "__main__":
    main()
