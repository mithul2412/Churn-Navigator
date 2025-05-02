import os
import mlflow
import mlflow.sklearn
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set MLflow tracking URI - can be local or remote
mlflow.set_tracking_uri("file:///home/stlp/mlflow")
mlflow.set_experiment("Churn Prediction")

# Load the transformed data
data_path = "/home/stlp/spark_output/transformed_churn_data"
print(f"Loading data from {data_path}")
table = pq.read_table(data_path)
df = table.to_pandas()

# Extract features and label
print("Preparing features and target")

# Convert dictionary features to numpy arrays
def extract_features(feature_dict):
    # For sparse vectors, we'll create a dense representation
    # This assumes your feature vector has a fixed size (44 in your case)
    size = 44  # Based on your data
    features = np.zeros(size)
    
    # Check if it's a dictionary or already a numeric type
    if isinstance(feature_dict, dict):
        # Get indices and values from the sparse vector
        indices = feature_dict.get('indices', [])
        values = feature_dict.get('values', []) if 'values' in feature_dict else [1.0] * len(indices)
        
        # Fill the dense vector
        for idx, val in zip(indices, values):
            if idx < size:
                features[idx] = val
    else:
        # If it's already numeric, just return it
        return feature_dict
        
    return features

# Apply the feature extraction to each row
X = np.array([extract_features(feat) for feat in df["features"]])
y = df["label"].values
customer_ids = df["customerID"].values

# Split data
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, customer_ids, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run(run_name="RandomForest_Churn_Model"):
    # Train model
    print("Training Random Forest model")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model
    mlflow.sklearn.log_model(rf, "random_forest_model")
    
    # Save predictions for sample customers
    results_df = pd.DataFrame({
        'customerID': ids_test,
        'actual_churn': y_test,
        'predicted_churn': y_pred,
        'churn_probability': y_prob
    })
    results_path = "/home/stlp/mlflow/predictions.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)  # Ensure directory exists
    results_df.head(100).to_csv(results_path, index=False)
    mlflow.log_artifact(results_path)
    
    print(f"Model training completed with accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")