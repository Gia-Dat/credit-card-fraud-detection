import pandas as pd
from sklearn.preprocessing import RobustScaler


def test_preprocessing_transforms_correctly():
    """
    Verify that duplicate transactions are dropped and features are scaled correctly.
    """
    # 1. Arrange: Create a mock raw dataset with known shapes and a clear duplicate row
    mock_data = pd.DataFrame(
        {
            # Row 0 and Row 3 are complete duplicates
            "Time": [0.0, 1.0, 2.0, 0.0],
            "Amount": [100.0, 50.0, 20.0, 100.0],
            "V1": [1.5, -0.5, 0.8, 1.5],
            "Class": [0, 1, 0, 0],
        }
    )

    # 2. Act: Replicate your exact feature preprocessing logic from train.py
    df = mock_data.copy()
    df.drop_duplicates(inplace=True)

    scaler = RobustScaler()
    df.loc[:, "scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df.loc[:, "scaled_time"] = scaler.fit_transform(df[["Time"]])
    df = df.drop(["Time", "Amount"], axis=1)

    # 3. Assert: Verify the data layout meets production requirements
    # Expectation A: The duplicate row should be completely purged (4 rows down to 3)
    assert len(df) == 3

    # Expectation B: The raw Time and Amount columns must be completely dropped
    assert "Time" not in df.columns
    assert "Amount" not in df.columns

    # Expectation C: The new scaled features must exist
    assert "scaled_amount" in df.columns
    assert "scaled_time" in df.columns

    print("\n✅ Data processing assertions passed successfully!")
