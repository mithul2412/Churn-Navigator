from datetime import datetime, timedelta
from pathlib import Path
import os
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(os.getenv("CHURN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "spark_transformation.py"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "transformed_churn_data.parquet"


default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def run_spark_transformation() -> str:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--output-path",
        str(OUTPUT_PATH),
    ]
    subprocess.run(cmd, check=True)
    return "Spark transformation completed successfully"


with DAG(
    "churn_spark_transformation",
    default_args=default_args,
    description="Transform churn data using Spark",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    transform_features = PythonOperator(
        task_id="transform_features",
        python_callable=run_spark_transformation,
    )
