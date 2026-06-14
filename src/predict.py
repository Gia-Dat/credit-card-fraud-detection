import numpy as np
from xgboost import XGBClassifier


def load_prediction_engine(model_path='models/xgboost_fraud_model.json'):
    """Loads the serialized model artifact back into memory."""
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def predict_transaction(model, feature_vector):
    """Generates an explicit prediction class and probability score."""
    data = np.array(feature_vector).reshape(1, -1)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return prediction, probability


if __name__ == "__main__":
    print("Loading operational fraud detection engine...")
    detector = load_prediction_engine()

    # Simulating a mock incoming card transaction with 30 preprocessed features (V1-V28, scaled_amount, scaled_time)
    mock_incoming_tx = [-2.31, 1.95, -1.60, 3.99, -0.52, -1.42, -2.53, 1.39, -2.77, -2.77,
                        3.20, -2.89, -0.60, -4.28, 0.38, -1.14, -2.83, -0.01, 0.41, 0.12,
                        0.51, -0.03, -0.46, 0.32, 0.04, 0.17, 0.26, -0.14, -0.30, -0.99]

    # Run prediction inference
    prediction, probability = predict_transaction(detector, mock_incoming_tx)

    print("\n==================== REAL-TIME FRAUD EVALUATION ====================")
    if prediction == 1:
        print(f"REJECTED - High risk anomaly flagged!")
        print(f"Risk Confidence Level: {probability:.2%}")
    else:
        print(f"APPROVED - Transaction verified legitimate.")
        print(f"Calculated Fraud Risk Profile: {probability:.4%}")
