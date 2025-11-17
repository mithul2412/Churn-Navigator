import argparse

import mlflow

try:
    from config import (
        MLFLOW_EXPERIMENT,
        MLFLOW_TRACKING_URI,
        MODEL_ARTIFACT_PATH,
        MODEL_NAME,
    )
except ImportError:
    from scripts.config import (
        MLFLOW_EXPERIMENT,
        MLFLOW_TRACKING_URI,
        MODEL_ARTIFACT_PATH,
        MODEL_NAME,
    )


def register_best_model(metric: str = "metrics.roc_auc") -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{MLFLOW_EXPERIMENT}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{metric} DESC", "attribute.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{MLFLOW_EXPERIMENT}'")

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    print(f"Registered model '{MODEL_NAME}' version {registered.version} from run {run_id}")
    return str(registered.version)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register best MLflow run as a named model")
    parser.add_argument(
        "--metric",
        default="metrics.roc_auc",
        help="Metric key used for selecting best run (default: metrics.roc_auc)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_best_model(metric=args.metric)


if __name__ == "__main__":
    main()
