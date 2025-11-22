from datetime import datetime, timedelta
from pathlib import Path
import os
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(os.getenv("CHURN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "transformed_churn_data.parquet"


default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _run_script(script_name: str, extra_args=None) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def run_spark_transformation() -> str:
    _run_script("spark_transformation.py", ["--output-path", str(OUTPUT_PATH)])
    return "Spark transformation completed successfully"


def run_mlflow_training() -> str:
    _run_script("mlflow_model.py", ["--data-path", str(OUTPUT_PATH)])
    return "MLflow model training completed successfully"


def register_best_model() -> str:
    _run_script("register_mlflow_model.py")
    return "Model registered successfully"


with DAG(
    "churn_prediction_pipeline",
    default_args=default_args,
    description="End-to-end churn prediction pipeline",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    transform_features = PythonOperator(
        task_id="transform_features",
        python_callable=run_spark_transformation,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_mlflow_training,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_best_model,
    )

    transform_features >> train_model >> register_model
