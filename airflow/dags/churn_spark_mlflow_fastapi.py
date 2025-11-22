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
LOG_DIR = PROJECT_ROOT / "logs"


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


def start_fastapi_service() -> str:
    existing = subprocess.run(
        ["pgrep", "-f", "uvicorn.*scripts.churn_api:app"],
        capture_output=True,
        text=True,
        check=False,
    )
    if existing.returncode == 0 and existing.stdout.strip():
        return "FastAPI service is already running"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_log = (LOG_DIR / "fastapi.out.log").open("a")
    err_log = (LOG_DIR / "fastapi.err.log").open("a")

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "scripts.churn_api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=out_log,
        stderr=err_log,
        preexec_fn=os.setpgrp,
    )

    return "FastAPI service started"


with DAG(
    "churn_prediction_pipeline_with_api",
    default_args=default_args,
    description="End-to-end churn pipeline with API startup",
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

    start_api = PythonOperator(
        task_id="start_api_service",
        python_callable=start_fastapi_service,
    )

    transform_features >> train_model >> register_model >> start_api
