from datetime import datetime, timedelta
from pathlib import Path
import os

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

PROJECT_ROOT = Path(os.getenv("CHURN_PROJECT_ROOT", "/opt/churn-navigator"))
ETL_SCRIPT = PROJECT_ROOT / "etl" / "feature_engineering.py"

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://admin:changeme@mongo:27017/?authSource=admin",
)
MONGO_DB = os.getenv("MONGO_DB", "churn")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "customer_churn")


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="feature_engineering",
    default_args=default_args,
    description="Load churn data from MongoDB and generate train/test parquet features",
    schedule="@daily",
    start_date=datetime(2025, 4, 1),
    catchup=False,
    max_active_runs=1,
) as dag:
    spark_feature_engineering = SparkSubmitOperator(
        task_id="spark_feature_engineering",
        conn_id="spark_default",
        application=str(ETL_SCRIPT),
        application_args=[
            "--mongo-uri",
            MONGO_URI,
            "--mongo-db",
            MONGO_DB,
            "--mongo-collection",
            MONGO_COLLECTION,
            "--train-out",
            "/data/train_features.parquet",
            "--test-out",
            "/data/test_features.parquet",
        ],
    )

    spark_feature_engineering
