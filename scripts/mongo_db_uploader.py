import pandas as pd
import pymongo
import json
from pymongo import MongoClient
import os
import glob

def connect_to_mongodb(connection_string, database_name):
    """
    Connect to MongoDB and return database object
    """
    try:
        # Create a connection to MongoDB
        client = MongoClient(connection_string)
        
        # Access the specified database
        db = client[database_name]
        
        print(f"Successfully connected to MongoDB database: {database_name}")
        return db
    
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        return None

def csv_to_mongodb(csv_file, db, collection_name, batch_size=1000):
    """
    Import CSV file to MongoDB collection
    
    Args:
        csv_file (str): Path to CSV file
        db: MongoDB database object
        collection_name (str): Name of the collection to import data into
        batch_size (int): Number of records to insert in each batch
    """
    try:
        # Check if collection exists, if so, ask for confirmation to proceed
        if collection_name in db.list_collection_names():
            confirmation = input(f"Collection '{collection_name}' already exists. Do you want to add to it? (y/n): ")
            if confirmation.lower() != 'y':
                print("Import cancelled.")
                return False
        
        # Create or access the collection
        collection = db[collection_name]
        
        # Read the CSV file
        print(f"Reading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Convert DataFrame to list of dictionaries (records)
        records = df.to_dict(orient='records')
        
        # Count of records
        total_records = len(records)
        print(f"Found {total_records} records to import")
        
        # Insert records in batches
        for i in range(0, total_records, batch_size):
            batch = records[i:i + batch_size]
            collection.insert_many(batch)
            print(f"Imported batch {i//batch_size + 1} ({min(i + batch_size, total_records)}/{total_records} records)")
        
        print(f"Successfully imported {total_records} records to collection '{collection_name}'")
        return True
    
    except Exception as e:
        print(f"Error importing CSV to MongoDB: {str(e)}")
        return False

def import_folder(folder_path, db, collection_prefix=""):
    """
    Import all CSV files from a folder to MongoDB
    
    Args:
        folder_path (str): Path to folder containing CSV files
        db: MongoDB database object
        collection_prefix (str): Prefix to add to collection names
    """
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to import")
    
    # Import each CSV file
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Create collection name from file name
        collection_name = f"{collection_prefix}{name_without_ext}" if collection_prefix else name_without_ext
        
        print(f"\nImporting {base_name} to collection '{collection_name}'...")
        csv_to_mongodb(csv_file, db, collection_name)

def main():
    # MongoDB connection settings
    connection_string = input("Enter MongoDB connection string (default: mongodb://localhost:27017): ") or "mongodb://localhost:27017"
    database_name = input("Enter database name: ")
    
    # Connect to MongoDB
    db = connect_to_mongodb(connection_string, database_name)
    if db is None:
        return
    
    # User choice for single file or folder
    choice = input("Import a single CSV file (1) or all CSV files in a folder (2)? ")
    
    if choice == "1":
        # Single file
        csv_file = input("Enter path to CSV file: ")
        collection_name = input("Enter collection name (default: use filename): ")
        
        if not collection_name:
            # Use filename without extension as collection name
            collection_name = os.path.splitext(os.path.basename(csv_file))[0]
            
        csv_to_mongodb(csv_file, db, collection_name)
    
    elif choice == "2":
        # Folder
        folder_path = input("Enter path to folder containing CSV files: ")
        collection_prefix = input("Enter prefix for collection names (optional): ")
        
        import_folder(folder_path, db, collection_prefix)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()