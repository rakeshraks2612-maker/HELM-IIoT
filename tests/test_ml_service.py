import numpy as np
import pandas as pd
from edge_gateway.schemas import PredictionRequest
from ml_inference_service.online_cluster import OnlineRegimeProfiler
from ml_inference_service.drift_detector import TelemetryDriftDetector
from ml_inference_service.train_pipeline import train_time_series_model, generate_time_series_dataset


def test_online_regime_profiler():
    profiler = OnlineRegimeProfiler()
    # Nominal
    cid_nom, name_nom = profiler.predict_regime(50.0, 0.1, 25.0, 41.0)
    assert cid_nom == 0
    assert "Nominal" in name_nom

    # Critical Anomaly
    cid_anom, name_anom = profiler.predict_regime(15.0, 4.5, 95.0, 75.0)
    assert cid_anom == 2
    assert "Anomaly" in name_anom


def test_drift_detector_psi():
    detector = TelemetryDriftDetector(psi_threshold=0.20)
    base = np.random.normal(50.0, 5.0, 300)
    shifted = np.random.normal(90.0, 5.0, 300)  # Significant shift

    psi_clean = detector.calculate_psi(base, base)
    assert psi_clean < 0.05

    psi_drift = detector.calculate_psi(base, shifted)
    assert psi_drift > 0.20


def test_time_series_training_pipeline(tmp_path):
    df = generate_time_series_dataset(n_samples=200)
    model_file = str(tmp_path / "test_engine.json")
    metrics = train_time_series_model(df=df, output_path=model_file)
    assert "mean_rmse" in metrics
    assert metrics["mean_rmse"] > 0.0
    assert len(metrics["cv_scores"]) == 5
