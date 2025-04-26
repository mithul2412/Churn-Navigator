import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np # Required by model/pipeline

# --- Configuration ---
# Set MLFLOW_TRACKING_URI environment variable before running the app
# e.g., export MLFLOW_TRACKING_URI='http://localhost:8080'
# OR set it here (less flexible):
# mlflow.set_tracking_uri("http://localhost:8080")

# Model details (replace with your registered model name and version/stage)
MODEL_NAME = "XGboost"
MODEL_VERSION = "1" # Or use a stage like "Production" or "Staging"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

# --- Pydantic Model for Input Data ---
# Define based on the original columns BEFORE preprocessing (excluding customerID and Churn)
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float # Input as float, was handled in training preprocessing

# --- FastAPI App Initialization ---
app = FastAPI(title="Churn Predictor API", version="1.0")

# --- Load MLflow Model ---
# Load the model pipeline once when the application starts
try:
    print(f"Loading model from URI: {MODEL_URI}")
    # Ensure MLFLOW_TRACKING_URI is set correctly for this line!
    model_pipeline = mlflow.pyfunc.load_model(MODEL_URI)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if model loading fails - crucial for API startup
    model_pipeline = None # Indicate model is not loaded
    # Consider raising an exception or handling this more robustly
    raise RuntimeError(f"Could not load MLflow model from {MODEL_URI}") from e


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Churn Predictor API is running. Use the /predict endpoint."}

@app.post("/predict")
async def predict_churn(features: CustomerFeatures):
    """Predicts churn probability for a single customer."""
    if model_pipeline is None:
         raise HTTPException(status_code=503, detail="Model not loaded. Service Unavailable.")

    try:
        # Convert input data to pandas DataFrame matching training structure
        # The pipeline's preprocessor expects specific column names and order
        input_df = pd.DataFrame([features.model_dump()])

        # Ensure TotalCharges is float, even if not strictly needed by model
        # (preprocessing in pipeline handles it, but good practice for input)
        input_df['TotalCharges'] = input_df['TotalCharges'].astype(float)

        print(f"Received input data:\n{input_df.to_string()}")

        try:
            # Make prediction using the standard .predict() method of pyfunc
            # This typically returns the predicted class (0 or 1 for classifiers)
            prediction_binary = model_pipeline.predict(input_df)
            # prediction_binary is likely a numpy array, e.g., array([0]) or array([1])
            # Get the first element and ensure it's a standard Python integer
            churn_prediction_class = int(prediction_binary[0])

            print(f"Predicted churn class: {churn_prediction_class}")  # Log the class

            # Return prediction result (binary class)
            return {"churn_prediction_class": churn_prediction_class} 
            # Return class instead of probability

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Optional: Add endpoint to check model status
@app.get("/health")
async def health_check():
    if model_pipeline:
        return {"status": "ok", "model_uri": MODEL_URI}
    else:
        return {"status": "error", "detail": "Model not loaded"}