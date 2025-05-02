from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import subprocess
import sys

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_spark_transformation(**kwargs):
    """
    Function to run Spark transformation script with proper error handling
    """
    # Set environment variables
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'
    script_path = "/mnt/c/Users/stlp/Desktop/VSCode/Churn_Navi_trial/scripts/spark_transformation.py"
    
    # Use subprocess to run the script and capture output
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Capture output in real-time
    stdout, stderr = process.communicate()
    
    # Log output
    if stdout:
        print(f"STDOUT: {stdout}")
    
    # Check for errors
    if process.returncode != 0:
        print(f"ERROR: {stderr}")
        raise Exception(f"Spark transformation failed with return code {process.returncode}")
    
    return "Spark transformation completed successfully"

with DAG(
    'churn_spark_transformation',
    default_args=default_args,
    description='Transform churn data using Spark',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    # Spark transformation task
    transform_features = PythonOperator(
        task_id='transform_features',
        python_callable=run_spark_transformation,
    )