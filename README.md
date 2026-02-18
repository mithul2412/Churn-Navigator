# Churn Navigator: Churn Analysis and Engagement

Churn Navigator is an end-to-end churn prediction project that combines MongoDB, Spark, Airflow, MLflow, FastAPI, Docker, and n8n.

## Project Structure

- `data/Churn_dataset.csv`: source dataset
- `scripts/mongo_db_uploader.py`: upload and clean CSV data into MongoDB
- `scripts/spark_transformation.py`: Spark feature transformation from MongoDB to Parquet
- `scripts/mlflow_model.py`: model training + MLflow logging
- `scripts/register_mlflow_model.py`: register best MLflow run
- `scripts/churn_api.py`: FastAPI inference service
- `scripts/config.py`: shared environment configuration
- `etl/feature_engineering.py`: Spark ETL job generating train/test parquet outputs
- `airflow/dags/*.py`: orchestration DAGs
- `n8n-custom/workflows/churn_high_risk_notification.json`: sample workflow export
- `docker-compose.yml`: local stack orchestration

## Data Source

- Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)

## Local Setup

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy and edit environment config:

```bash
cp .env.example .env
```

4. Verify key values in `.env`:
- `MONGO_URI`
- `MONGO_DB`
- `MONGO_COLLECTION`
- `SPARK_OUTPUT_PATH`
- `MLFLOW_TRACKING_URI`

## Run Pipeline Scripts (Without Airflow)

1. Upload raw data to MongoDB:

```bash
python scripts/mongo_db_uploader.py --drop-existing
```

2. Build transformed feature parquet with Spark:

```bash
python scripts/spark_transformation.py
```

3. Train model and log artifacts to MLflow:

```bash
python scripts/mlflow_model.py
```

4. Register best run as model:

```bash
python scripts/register_mlflow_model.py
```

5. Start API server:

```bash
python scripts/churn_api.py
```

API endpoints:
- `GET /health`
- `POST /predict`

Sample request body:

```json
{
  "customer_id": "7590-VHVEG",
  "features": [0.0, 1.0, 0.0, 0.0, 1.0]
}
```

Note: feature vector length must match the trained model input dimension.

## Run with Docker Compose

```bash
docker compose up --build
```

Services:
- MongoDB: `localhost:27017`
- Spark master UI: `localhost:8081`
- Airflow UI: `localhost:8080` (admin/admin)
- n8n: `localhost:5678` (admin/changeme)

## Airflow DAGs

- `churn_data_upload`: uploads source CSV into MongoDB
- `churn_spark_transformation`: Spark feature transformation
- `churn_prediction_pipeline`: transform -> train -> register
- `churn_prediction_pipeline_with_api`: transform -> train -> register -> start API
- `feature_engineering`: SparkSubmit ETL job writing `/data/train_features.parquet` and `/data/test_features.parquet`

## n8n Workflow

Import `n8n-custom/workflows/churn_high_risk_notification.json` inside n8n to bootstrap a high-risk churn alert flow.

## License

This project is licensed under the MIT License.
