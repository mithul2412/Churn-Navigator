import pandas as pd
from pymongo import MongoClient
import os

# Define file path - update this to your actual path
DATA_PATH = "data/Churn_dataset.csv"

def upload_to_mongodb():
    try:
        # Replace this with your MongoDB Atlas connection string
        connection_string = "mongodb+srv://churn:churn@telco.l9welj0.mongodb.net/"
        
        # Connect to MongoDB Atlas
        client = MongoClient(connection_string)
        
        # Create or access database
        db = client['churn']
        
        # Create or access collection
        collection = db['customer_churn']
        
        # Check if CSV file exists
        if not os.path.exists(DATA_PATH):
            print(f"Error: File not found at {DATA_PATH}")
            return
            
        # Load CSV file
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        
        # Convert SeniorCitizen from 0/1 to No/Yes for consistency
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        # Convert TotalCharges to float (handling any non-numeric values)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Insert data into MongoDB
        print(f"Uploading {len(records)} records to MongoDB Atlas...")
        result = collection.insert_many(records)
        
        print(f"Upload complete. Inserted {len(result.inserted_ids)} documents.")
        
        # Create indexes for common query fields
        collection.create_index("customerID")
        collection.create_index("Churn")
        
        print("Created indexes on customerID and Churn fields.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    upload_to_mongodb()
