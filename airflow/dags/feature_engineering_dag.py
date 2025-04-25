# airflow/dags/feature_engineering_dag.py

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# default_args will apply to all tasks in this DAG
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
    description="Load raw churn data from MongoDB, run Spark feature engineering, write out Parquet",
    schedule_interval="@daily",           # adjust to your needs
    start_date=datetime(2025, 4, 1),     # or use days_ago(1)
    catchup=False,
    max_active_runs=1,
) as dag:

    spark_feature_engineering = SparkSubmitOperator(
        task_id="spark_feature_engineering",
        conn_id="spark_default",
        application="/etl/feature_engineering.py",
        # if your script takes any args, you can pass them here:
        application_args=[
            # for example, you could parse these in your script via argparse
            "--mongo-uri", "mongodb://admin:changeme@mongo:27017/churndb.raw_churn_data?authSource=admin",
            "--train-out", "/data/train_features.parquet",
            "--test-out",  "/data/test_features.parquet",
        ],
        # spark-defaults (jars, packages) are already set via SPARK_PACKAGES in your
        # docker-compose, so no need to repeat them here.
    )

    spark_feature_engineering  # this is the only task in the DAG for now
