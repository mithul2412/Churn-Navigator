from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
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

def run_mlflow_training(**kwargs):
    """
    Function to run MLflow model training with proper error handling
    """
    script_path = "/mnt/c/Users/stlp/Desktop/VSCode/Churn_Navi_trial/scripts/mlflow_model.py"
    
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
        raise Exception(f"MLflow training failed with return code {process.returncode}")
    
    return "MLflow model training completed successfully"

def register_best_model(**kwargs):
    script_path = "/mnt/c/Users/stlp/Desktop/VSCode/Churn_Navi_trial/scripts/register_mlflow_model.py"
    
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    
    if stdout:
        print(f"STDOUT: {stdout}")
    
    if process.returncode != 0:
        print(f"ERROR: {stderr}")
        raise Exception(f"Model registration failed with return code {process.returncode}")
    
    return "Model registered successfully"

with DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end churn prediction pipeline',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    # Spark transformation task
    transform_features = PythonOperator(
        task_id='transform_features',
        python_callable=run_spark_transformation,
    )
    
    # MLflow model training task
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=run_mlflow_training,
    )
    
    # Register best model
    register_model = PythonOperator(
        task_id='register_model',
        python_callable=register_best_model,
    )
    
    # Start MLflow UI (optional)
    start_mlflow_ui = BashOperator(
        task_id='start_mlflow_ui',
        bash_command='mlflow ui --backend-store-uri file:///home/stlp/mlflow --port 5000 --host 0.0.0.0 &',
    )
    
    # Set dependencies
    transform_features >> train_model >> register_model >> start_mlflow_ui
