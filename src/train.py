import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import mlflow
import mlflow.xgboost


def run_training_pipeline():
    mlflow.set_experiment("Credit_Card_Fraud_Detection")

    print("Loading dataset...")
    df = pd.read_csv('data/creditcard.csv')

    print("Removing duplicate transactions...")
    df.drop_duplicates(inplace=True)

    print("Scaling Time and Amount using RobustScaler...")
    scaler = RobustScaler()
    df.loc[:, 'scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df.loc[:, 'scaled_time'] = scaler.fit_transform(df[['Time']])
    df = df.drop(['Time', 'Amount'], axis=1)

    print("Splitting data...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    params = {
        "learning_rate": 0.1,
        "max_depth": 6,
        "scale_pos_weight": 1,
        "eval_metric": 'logloss',
        "random_state": 42,
        "n_jobs": -1
    }

    print("Training XGBoost with MLflow tracking turned on...")

    # Start a new tracking session block
    with mlflow.start_run(run_name="XGBoost_Local_Baseline"):

        # Log your hyperparameter settings automatically
        mlflow.log_params(params)

        # Train the model using the dictionary parameters
        xgb = XGBClassifier(**params)
        xgb.fit(X_train, y_train)

        # Calculate your performance metrics
        y_proba = xgb.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log your scores to the dashboard ledger
        mlflow.log_metric("PR-AUC", pr_auc)
        mlflow.log_metric("ROC-AUC", roc_auc)

        # Save the actual model binary artifact directly inside MLflow
        mlflow.xgboost.log_model(xgb, artifact_path="fraud_model")

        # Your standard local backup export as a fallback
        os.makedirs('models', exist_ok=True)
        model_path = 'models/xgboost_fraud_model.json'
        print(f"Exporting local model artifact backup to: {model_path}...")
        xgb.save_model(model_path)

    print(f"Success! Final Tracked PR-AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    run_training_pipeline()
