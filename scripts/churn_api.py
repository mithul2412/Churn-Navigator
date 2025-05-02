import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import Dict, Optional

# Define the input data model
class CustomerFeatures(BaseModel):
    features: Dict

class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    churn_prediction: bool
    model_version: str

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)

# Define the model
model = None
model_version = None

# Load the model on startup
@app.on_event("startup")
def load_model():
    global model, model_version
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:///home/stlp/mlflow")
    
    # Load the registered model
    try:
        # Try loading the latest production version
        model = mlflow.sklearn.load_model("models:/churn_prediction_model/latest")
        model_info = mlflow.register_model.get_latest_versions("churn_prediction_model", stages=["None"])[0]
        model_version = model_info.version
    except Exception as e:
        print(f"Error loading registered model: {e}")
        # Fall back to a specific run
        latest_run_id = mlflow.search_runs(experiment_names=["Churn Prediction"]).iloc[0].run_id
        model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/random_forest_model")
        model_version = "dev"
    
    print(f"Model loaded successfully. Version: {model_version}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract features from input
        features = np.zeros(44)  # Assuming 44 features from your transformed data
        
        # Fill features array with values from input
        for idx, value in data.features.items():
            idx_int = int(idx)
            if idx_int < 44:
                features[idx_int] = value
        
        # Make prediction
        features_reshaped = features.reshape(1, -1)
        churn_probability = model.predict_proba(features_reshaped)[0, 1]
        churn_prediction = churn_probability >= 0.5
        
        return PredictionResponse(
            churn_probability=float(churn_probability),
            churn_prediction=bool(churn_prediction),
            model_version=str(model_version)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("churn_api:app", host="0.0.0.0", port=8000, reload=True)