# src/train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


def run_training_pipeline():
    # 1. Load dataset
    print("Loading dataset...")
    df = pd.read_csv('data/creditcard.csv')

    # 2. Clean dataset
    print("Removing duplicate transactions...")
    df.drop_duplicates(inplace=True)

    # 3. Scale for unbalanced dataset
    print("Scaling Time and Amount using RobustScaler...")
    scaler = RobustScaler()
    df.loc[:, 'scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df.loc[:, 'scaled_time'] = scaler.fit_transform(df[['Time']])

    df = df.drop(['Time', 'Amount'], axis=1)

    # 4. Split dataset
    print("Splitting data...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Train model with XGBoost hyperparameters found in notebook
    print("Training...")
    xgb = XGBClassifier(
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=1,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)

    # Creating the target directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgboost_fraud_model.json'

    # 6. Export model
    print(f"Exporting model artifact to: {model_path}...")
    xgb.save_model(model_path)
    print("Success")


if __name__ == "__main__":
    run_training_pipeline()
