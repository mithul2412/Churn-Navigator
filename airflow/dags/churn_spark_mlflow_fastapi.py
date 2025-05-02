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
    """
    Function to register the best model in MLflow
    """
    import mlflow
    import mlflow.sklearn
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:///home/stlp/mlflow")
    
    # Get the best run
    experiment = mlflow.get_experiment_by_name("Churn Prediction")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) > 0:
        # Sort runs by metric (e.g., ROC AUC)
        best_run = runs.sort_values("metrics.roc_auc", ascending=False).iloc[0]
        run_id = best_run.run_id
        
        # Register the model
        model_uri = f"runs:/{run_id}/random_forest_model"
        mv = mlflow.register_model(model_uri, "churn_prediction_model")
        
        print(f"Model registered: churn_prediction_model version {mv.version}")
    else:
        print("No runs found for the experiment")
        raise Exception("No runs found, cannot register model")
    
    return f"Model registered successfully: version {mv.version}"

def start_fastapi_service(**kwargs):
    """
    Function to start the FastAPI service
    """
    # Check if service is already running
    process = subprocess.Popen(
        ["pgrep", "-f", "churn_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, _ = process.communicate()
    
    if stdout.strip():
        print("FastAPI service is already running")
        return "FastAPI service is already running"
    
    # Start the service in the background
    script_path = "/mnt/c/Users/stlp/Desktop/VSCode/Churn_Navi_trial/scripts/churn_api.py"
    
    # Use nohup to keep the process running after Airflow task completes
    subprocess.Popen(
        ["nohup", sys.executable, script_path, "&"],
        stdout=open('/home/stlp/fastapi.log', 'w'),
        stderr=open('/home/stlp/fastapi.err', 'w'),
        preexec_fn=os.setpgrp
    )
    
    print("FastAPI service started")
    return "FastAPI service started"

with DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end churn prediction pipeline',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    # Upload data to MongoDB (assuming this is already done)
    
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
    
    # Start MLflow UI
    start_mlflow_ui = BashOperator(
        task_id='start_mlflow_ui',
        bash_command='pgrep -f "mlflow ui" || mlflow ui --backend-store-uri file:///home/stlp/mlflow --port 5000 --host 0.0.0.0 > /home/stlp/mlflow_ui.log 2>&1 &',
    )
    
    # Start FastAPI service
    start_api = PythonOperator(
        task_id='start_api_service',
        python_callable=start_fastapi_service,
    )
    
    # Set dependencies
    transform_features >> train_model >> register_model >> start_mlflow_ui >> start_api