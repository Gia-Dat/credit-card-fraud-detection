# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from xgboost import XGBClassifier

# 1. Initialize the FastAPI instance
app = FastAPI(
    title="Credit Card Fraud Detection Engine",
    description="Production-grade API for real-time transaction screening.",
    version="1.0.0"
)

# 2. Load the exported machine learning model artifact into memory
MODEL_PATH = "models/xgboost_fraud_model.json"
model = XGBClassifier()
model.load_model(MODEL_PATH)

# 3. Define the structured layout for incoming data using Pydantic


class Transaction(BaseModel):
    # Expecting an array of exactly 30 numerical values (V1-V28, scaled_amount, scaled_time)
    features: List[float] = Field(
        ...,
        example=[-2.31, 1.95, -1.60, 3.99, -0.52, -1.42, -2.53, 1.39, -2.77, -2.77,
                 3.20, -2.89, -0.60, -4.28, 0.38, -1.14, -2.83, -0.01, 0.41, 0.12,
                 0.51, -0.03, -0.46, 0.32, 0.04, 0.17, 0.26, -0.14, -0.30, -0.99]
    )


@app.get("/")
def health_check():
    """Simple heart-beat route to verify the API is running smoothly."""
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_transaction_fraud(transaction: Transaction):
    """Accepts a feature vector and returns an instantaneous fraud risk assessment."""

    # Validate feature vector length
    if len(transaction.features) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feature count. Expected exactly 30 features, received {len(transaction.features)}."
        )

    try:
        # Convert incoming list into a 2D numpy array for the model
        data_vector = np.array(transaction.features).reshape(1, -1)

        # Run inference
        prediction = int(model.predict(data_vector)[0])
        probability = float(model.predict_proba(data_vector)[0][1])

        # Build structured JSON payload response
        return {
            "is_fraud": prediction == 1,
            "fraud_probability": round(probability, 6),
            "status": "REJECTED" if prediction == 1 else "APPROVED"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference error: {str(e)}")
