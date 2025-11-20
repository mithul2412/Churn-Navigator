from datetime import datetime, timedelta
from pathlib import Path
import os
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(os.getenv("CHURN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "mongo_db_uploader.py"
DATA_PATH = PROJECT_ROOT / "data" / "Churn_dataset.csv"


default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def upload_churn_data() -> str:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--data-path",
        str(DATA_PATH),
        "--drop-existing",
    ]
    subprocess.run(cmd, check=True)
    return "Churn data uploaded to MongoDB"


with DAG(
    "churn_data_upload",
    default_args=default_args,
    description="Upload churn dataset to MongoDB",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    upload_task = PythonOperator(
        task_id="upload_churn_data",
        python_callable=upload_churn_data,
    )
