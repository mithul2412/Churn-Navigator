from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from pymongo import MongoClient
import os

# Define default arguments for the DAG
default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the path to your CSV file
DATA_PATH = "/mnt/c/Users/stlp/Desktop/VSCode/Churn_Navi_trial/data/Churn_dataset.csv"

def upload_churn_data_to_mongodb(**context):
    """
    Function to upload churn dataset to MongoDB Atlas
    """
    from airflow.hooks.base import BaseHook
    
    # Get MongoDB connection from Airflow connections
    conn = BaseHook.get_connection("mongodb_default")  # Use your connection ID
    
    # Create MongoDB connection string
    host = conn.host
    username = conn.login
    password = conn.password
    database = conn.schema if conn.schema else "churn"
    
    # Create MongoDB connection string
    connection_string = f"mongodb+srv://{username}:{password}@{host}/{database}"
    
    # Connect to MongoDB
    client = MongoClient(connection_string)
    db = client[database]
    collection = db["churn"]
    
    # Check if file exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")
    
    # Load and process CSV data
    df = pd.read_csv(DATA_PATH)
    
    # Convert SeniorCitizen from 0/1 to No/Yes for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Convert TotalCharges to float (handling any non-numeric values)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Convert DataFrame to dictionary records
    records = df.to_dict('records')
    
    # Check if collection already has data
    existing_count = collection.count_documents({})
    if existing_count > 0:
        print(f"Collection already contains {existing_count} documents. Dropping collection.")
        collection.delete_many({})
    
    # Insert data into MongoDB
    result = collection.insert_many(records)
    
    # Create indexes for common query fields
    collection.create_index("customerID")
    collection.create_index("churn")
    
    return f"Successfully uploaded {len(result.inserted_ids)} records to MongoDB Atlas."

# Create the DAG
with DAG(
    'churn_data_upload',
    default_args=default_args,
    description='Upload churn dataset to MongoDB Atlas',
    schedule=None,  # Changed from schedule_interval to schedule
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    # Task to upload data to MongoDB
    upload_task = PythonOperator(
        task_id='upload_churn_data',
        python_callable=upload_churn_data_to_mongodb,
    )