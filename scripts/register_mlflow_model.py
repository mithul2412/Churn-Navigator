import mlflow
import mlflow.sklearn
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:///home/stlp/mlflow")

# Get the best run (you can replace this with the specific run_id if you know it)
experiment = mlflow.get_experiment_by_name("Churn Prediction")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

if len(runs) > 0:
    # Sort runs by metric (e.g., ROC AUC)
    best_run = runs.sort_values("metrics.roc_auc", ascending=False).iloc[0]
    run_id = best_run.run_id
    
    # Register the model
    model_uri = f"runs:/{run_id}/random_forest_model"
    model_details = mlflow.register_model(model_uri, "churn_prediction_model")
    
    print(f"Model registered: {model_details.name} version {model_details.version}")
else:
    print("No runs found for the experiment")