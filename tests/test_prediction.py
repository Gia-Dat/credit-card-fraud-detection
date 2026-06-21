import numpy as np
import pytest

from src.predict import load_prediction_engine


def test_prediction_engine_valid_input():
    """
    Ensure the inference model correctly evaluates a complete 30-feature profile.
    """
    # 1. Arrange: Create a valid mock transaction vector (30 inputs)
    # 28 PCA features + scaled_amount + scaled_time
    valid_features = list(np.random.uniform(-2.0, 2.0, 28)) + [0.5, -0.1]

    # 2. Act: Run it through your inference module logic
    # (Assuming your src/predict.py exposes a function or model pipeline)
    model = load_prediction_engine()
    prediction_proba = model.predict_proba([valid_features])[0, 1]

    # 3. Assert: The model must return a clear probability score between 0 and 1
    assert 0.0 <= prediction_proba <= 1.0


def test_prediction_engine_invalid_dimensions():
    """
    Ensure the system rejects inputs that don't match the required 30 features.
    """
    # Arrange: Pass an incomplete vector (only 5 features instead of 30)
    broken_features = [1.5, -0.2, 0.5, 1.2, -0.9]
    model = load_prediction_engine()

    # Act & Assert: The execution should throw a ValueError or shape mismatch alert
    with pytest.raises(ValueError):
        model.predict([broken_features])
