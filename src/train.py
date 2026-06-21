import os

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

# ---   S3/ LocalStack routing configuration
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:4566"
os.environ["AWS_ACCESS_KEY_ID"] = "mock_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "mock_secret"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


def run_training_pipeline():
    # Define your virtual S3 storage route
    BUCKET_NAME = "mlflow-model-registry"
    artifact_uri = f"s3://{BUCKET_NAME}/mlflow-artifacts"

    experiment_name = "Credit_Card_Fraud_Pipeline"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_uri)

    mlflow.set_experiment(experiment_name)

    print("Loading dataset...")
    df = pd.read_csv("data/creditcard.csv")

    print("Removing duplicate transactions...")
    df.drop_duplicates(inplace=True)

    print("Scaling Time and Amount using RobustScaler...")
    scaler = RobustScaler()
    df.loc[:, "scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df.loc[:, "scaled_time"] = scaler.fit_transform(df[["Time"]])
    df = df.drop(["Time", "Amount"], axis=1)

    print("Splitting data...")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    params = {
        "learning_rate": 0.1,
        "max_depth": 6,
        "scale_pos_weight": 1,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    print("Training XGBoost with MLflow tracking turned on...")

    # Start a new tracking session block
    with mlflow.start_run():
        # Log your hyperparameter settings automatically
        mlflow.log_params(params)

        # Train the model using the dictionary parameters
        xgb = XGBClassifier(**params)
        xgb.fit(X_train, y_train)

        # Calculate your performance metrics
        y_proba = xgb.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log the model
        mlflow.xgboost.log_model(xgb, artifact_path="fraud_model")

        # Log the metrics
        mlflow.log_metric("PR-AUC", pr_auc)
        mlflow.log_metric("ROC-AUC", roc_auc)

        # Your standard local backup export as a fallback
        os.makedirs("models", exist_ok=True)
        model_path = "models/xgboost_fraud_model.json"
        print(f"Exporting local model artifact backup to: {model_path}...")
        xgb.save_model(model_path)

    print(f"Success! Final Tracked PR-AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    run_training_pipeline()
