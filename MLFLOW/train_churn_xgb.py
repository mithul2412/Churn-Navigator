import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import XGBoost
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn # We can still use mlflow.sklearn with the XGBoost scikit-learn wrapper

logging.basicConfig(level=logging.INFO) # Changed to INFO to see more logs
logger = logging.getLogger(__name__)


def eval_clf_metrics(actual, pred_proba, pred_binary):
    """Helper function to evaluate classification metrics."""
    try:
        accuracy = accuracy_score(actual, pred_binary)
        precision = precision_score(actual, pred_binary)
        recall = recall_score(actual, pred_binary)
        f1 = f1_score(actual, pred_binary)
        # Use predicted probabilities for AUC
        auc = roc_auc_score(actual, pred_proba)
        return accuracy, precision, recall, f1, auc
    except Exception as e:
        logger.warning(f"Could not calculate metrics: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)  # for reproducibility

    try:
        # --- 1. Load Data ---
        logger.info("Loading data...")
        df = pd.read_csv("Churn_dataset.csv")

        # --- 2. Basic Preprocessing ---
        logger.info("Preprocessing data...")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        df = df.drop("customerID", axis=1)

        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Corrected lines:
        categorical_cols = X.select_dtypes(include='object').columns.tolist() # Use keyword argument
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()  # Use keyword argument
        if 'SeniorCitizen' in categorical_cols:
            categorical_cols.remove('SeniorCitizen')
        if 'SeniorCitizen' not in numerical_cols:
             numerical_cols.append('SeniorCitizen')

        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"Numerical columns: {numerical_cols}")

        # --- 3. Define Preprocessing Steps ---
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # --- 4. Define Model ---
        # Using XGBoost Classifier
        # Common parameters:
        # n_estimators: number of trees
        # learning_rate: step size shrinkage
        # max_depth: maximum depth of a tree
        # use_label_encoder=False and eval_metric='logloss' are recommended for newer versions
        model = XGBClassifier(
            use_label_encoder=False, # Recommended for newer XGBoost versions
            eval_metric='logloss',   # Common choice for binary classification probability
            random_state=42,
            n_estimators=100,        # Example parameter
            learning_rate=0.1        # Example parameter
            )

        # --- 5. Create Full Pipeline ---
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )

        # --- 6. Start MLflow Run ---
        with mlflow.start_run() as run:
            run_id = run.info.run_uuid
            print(f"\nStarting MLflow Run: {run_id}")
            logger.info("Starting MLflow run...")

            # Log parameters
            mlflow.log_param("model_type", "XGBoostClassifier")
            # Log all parameters from the XGBoost model instance directly
            # This is simpler and captures defaults correctly.
            mlflow.log_params(model.get_params())
            print("Logged model parameters to MLflow.")

            # You could also log specific parameters like this if preferred:
            # n_estimators_val = model.get_params()['n_estimators']
            # learning_rate_val = model.get_params()['learning_rate']
            # eval_metric_val = model.get_params()['eval_metric']
            # mlflow.log_param("n_estimators", n_estimators_val)
            # mlflow.log_param("learning_rate", learning_rate_val)
            # mlflow.log_param("eval_metric", eval_metric_val)
            
            # Fit the pipeline
            logger.info("Training model pipeline...")
            pipeline.fit(X_train, y_train)

            # Make predictions
            logger.info("Evaluating model...")
            y_pred_binary = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            # Evaluate metrics
            (accuracy, precision, recall, f1, auc) = eval_clf_metrics(
                y_test, y_pred_proba, y_pred_binary
            )

            print("\nEvaluation Metrics:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            print(f"  AUC: {auc:.3f}")

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            print("\nLogged metrics to MLflow.")

            # Log the model pipeline
            # mlflow.sklearn works well with sklearn-compatible wrappers like XGBClassifier
            mlflow.sklearn.log_model(pipeline, "model_pipeline")
            print("Logged model pipeline artifact to MLflow.")

            print("-" * 20)
            print(f"MLflow Run completed. Check the UI at your tracking URI.")
            print(f"Run ID: {run_id}")
            print("-" * 20)

    except FileNotFoundError:
        logger.error(f"Error: Churn_dataset.csv not found. Please make sure it's in the directory.")
    except ModuleNotFoundError:
        logger.error("Error: xgboost library not found. Please install it using 'pip install xgboost'")
    except Exception as e:
        logger.error(f"An error occurred during the script execution: {e}", exc_info=True)