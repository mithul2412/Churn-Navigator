from typing import List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

try:
    from config import (
        API_HOST,
        API_PORT,
        CHURN_THRESHOLD,
        MLFLOW_EXPERIMENT,
        MLFLOW_TRACKING_URI,
        MODEL_ARTIFACT_PATH,
        MODEL_NAME,
    )
except ImportError:
    from scripts.config import (
        API_HOST,
        API_PORT,
        CHURN_THRESHOLD,
        MLFLOW_EXPERIMENT,
        MLFLOW_TRACKING_URI,
        MODEL_ARTIFACT_PATH,
        MODEL_NAME,
    )


class PredictionRequest(BaseModel):
    features: List[float]
    customer_id: Optional[str] = None


class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    churn_prediction: bool
    model_version: str


app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0",
)

model = None
model_version = "unknown"


def _load_model_from_registry():
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    loaded_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")

    try:
        latest_versions = client.get_latest_versions(MODEL_NAME)
        version = latest_versions[0].version if latest_versions else "latest"
    except Exception:
        version = "latest"

    return loaded_model, str(version)


def _load_model_from_latest_run():
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise ValueError(
            f"Experiment '{MLFLOW_EXPERIMENT}' not found and registry model is unavailable"
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError("No MLflow runs found for fallback model loading")

    run_id = runs.iloc[0]["run_id"]
    loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}")
    return loaded_model, f"run:{run_id}"


@app.on_event("startup")
def load_model() -> None:
    global model, model_version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model, model_version = _load_model_from_registry()
    except Exception:
        model, model_version = _load_model_from_latest_run()


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "model_loaded": model is not None, "model_version": model_version}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(payload: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.asarray(payload.features, dtype=float)
    expected_dim = getattr(model, "n_features_in_", None)
    if expected_dim is not None and features.shape[0] != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_dim} features, received {features.shape[0]}",
        )

    row = features.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        churn_probability = float(model.predict_proba(row)[0, 1])
    else:
        churn_probability = float(model.predict(row)[0])

    return PredictionResponse(
        customer_id=payload.customer_id,
        churn_probability=churn_probability,
        churn_prediction=churn_probability >= CHURN_THRESHOLD,
        model_version=model_version,
    )


@app.get("/")
def read_root() -> dict:
    return {"message": "Welcome to the Churn Prediction API", "model_version": model_version}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("churn_api:app", host=API_HOST, port=API_PORT, reload=False)
