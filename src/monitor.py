import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from fastapi import FastAPI
from prometheus_client import Gauge, make_asgi_app
from sklearn.preprocessing import RobustScaler

# 1. Initialize a mini API server specifically to serve metrics to Prometheus
monitor_app = FastAPI(title="Evidently AI")

# Add the standard Prometheus metrics endpoint route at /metrics
metrics_app = make_asgi_app()
monitor_app.mount("/metrics", metrics_app)

# 2. Define Prometheus Gauges to track data drift status
# (0 = No Drift, 1 = Significant Drift Detected)
DRIFT_STATUS_GAUGE = Gauge(
    "dataset_drift_status",
    "Overall data drift status (1 if drift detected, 0 otherwise)",
)
DRIFTED_FEATURES_GAUGE = Gauge(
    "drifted_features_count",
    "Total number of features showing significant statistical drift",
)

# Load your baseline training data to serve as the golden reference
# (We only load a subset of rows to keep the memory footprint light)
REFERENCE_DATA = pd.read_csv("data/creditcard.csv", nrows=5000)

# Replicate your train.py preprocessing on the reference data so it matches production layout
scaler = RobustScaler()
REFERENCE_DATA["scaled_amount"] = scaler.fit_transform(REFERENCE_DATA[["Amount"]])
REFERENCE_DATA["scaled_time"] = scaler.fit_transform(REFERENCE_DATA[["Time"]])
REFERENCE_DATA = REFERENCE_DATA.drop(["Time", "Amount"], axis=1)
# Rename 'Class' to 'prediction' to line up with production log labels
REFERENCE_DATA = REFERENCE_DATA.rename(columns={"Class": "prediction"})


@monitor_app.get("/evaluate")
def evaluate_drift():
    """
    Reads recent production transaction logs, calculates data drift
    against the baseline, and updates the Prometheus metric gauges.
    """
    try:
        # Load the production data collected by FastAPI
        production_data = pd.read_csv("data/production_logs.csv")

        if len(production_data) < 10:
            return {"status": "Waiting for more incoming data logs..."}

        # Initialize the Evidently Data Drift Report
        data_drift_preset = DataDriftPreset()
        data_drift_report = Report(metrics=[data_drift_preset])

        # Compare reference baseline vs current production records
        data_drift_report.run(
            reference_data=REFERENCE_DATA, current_data=production_data
        )

        metrics_summary = data_drift_preset.get_result()

        dataset_drifted = 1 if metrics_summary.dataset_drift else 0
        drifted_features = metrics_summary.number_of_drifted_columns

        # Update our Prometheus live tracking endpoints
        DRIFT_STATUS_GAUGE.set(dataset_drifted)
        DRIFTED_FEATURES_GAUGE.set(drifted_features)

        return {
            "status": "Success",
            "dataset_drift_detected": bool(dataset_drifted),
            "number_of_drifted_features": drifted_features,
        }

    except Exception as e:
        return {"status": "Error", "detail": str(e)}
