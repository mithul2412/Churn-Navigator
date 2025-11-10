import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _file_uri(path: Path) -> str:
    return f"file://{path.resolve().as_posix()}"


DATA_PATH = os.getenv("DATA_PATH", str(PROJECT_ROOT / "data" / "Churn_dataset.csv"))
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://admin:changeme@localhost:27017/?authSource=admin",
)
MONGO_DB = os.getenv("MONGO_DB", "churn")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "customer_churn")

SPARK_OUTPUT_PATH = os.getenv(
    "SPARK_OUTPUT_PATH",
    str(PROJECT_ROOT / "data" / "processed" / "transformed_churn_data.parquet"),
)
SPARK_JARS_PACKAGES = os.getenv(
    "SPARK_JARS_PACKAGES",
    "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    _file_uri(PROJECT_ROOT / "mlruns"),
)
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "Churn Prediction")
MODEL_NAME = os.getenv("MODEL_NAME", "churn_prediction_model")
MODEL_ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "random_forest_model")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))
