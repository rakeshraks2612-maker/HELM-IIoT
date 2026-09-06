"""
Production Training Pipeline with Time-Series Cross Validation.
Uses TimeSeriesSplit to prevent temporal data leakage and trains XGBoost regression models.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import structlog
from config.settings import settings

logger = structlog.get_logger("helm-train-pipeline")


def generate_time_series_dataset(n_samples: int = 1500) -> pd.DataFrame:
    """Generates a rich time-series dataset with realistic lag features and physical RTT latency."""
    np.random.seed(42)
    time_index = pd.date_range(start="2026-08-01 00:00:00", periods=n_samples, freq="1s")

    throughput = np.random.normal(52.0, 8.0, n_samples)
    packet_drop = np.random.exponential(0.35, n_samples)
    buffer_util = np.random.uniform(20.0, 75.0, n_samples)
    temp = np.random.normal(43.0, 2.5, n_samples)
    
    # Introduce temporal congestion waves scaled to sample size
    c1_start, c1_end = int(0.15 * n_samples), int(0.25 * n_samples)
    for i in range(c1_start, min(c1_end, n_samples)):
        packet_drop[i] += 2.8
        buffer_util[i] += 25.0
        temp[i] += 6.0

    c2_start, c2_end = int(0.55 * n_samples), int(0.65 * n_samples)
    for i in range(c2_start, min(c2_end, n_samples)):
        throughput[i] -= 22.0
        packet_drop[i] += 4.2
        buffer_util[i] += 30.0

    df = pd.DataFrame({
        "timestamp": time_index,
        "throughput_mbps": throughput,
        "packet_drop_percentage": packet_drop,
        "buffer_utilization_percentage": buffer_util,
        "node_temperature_celsius": temp
    })

    # Cluster regime assignment
    labels = []
    for d, b, t in zip(df["packet_drop_percentage"], df["buffer_utilization_percentage"], df["node_temperature_celsius"]):
        if d >= 2.5 or t >= 65.0:
            labels.append(2)
        elif b >= 65.0 or d >= 0.8:
            labels.append(1)
        else:
            labels.append(0)
    df["dynamic_operational_label"] = labels

    # Physical RTT target simulation
    base_rtt = 38.0 + (df["packet_drop_percentage"] * 10.5) + (df["buffer_utilization_percentage"] * 0.22) + (df["dynamic_operational_label"] * 8.0)
    jitter = np.random.normal(0, 1.2, size=n_samples)
    df["network_latency_ms"] = base_rtt + jitter

    # Lag features
    df["lag_latency_1"] = df["network_latency_ms"].shift(1).bfill()
    df["lag_latency_2"] = df["network_latency_ms"].shift(2).bfill()
    df["throughput_slope"] = df["throughput_mbps"].diff().fillna(0.0)
    df["buffer_peak"] = df["buffer_utilization_percentage"].rolling(10, min_periods=1).max()

    return df


def train_time_series_model(df: pd.DataFrame = None, output_path: str = None) -> dict:
    """Trains XGBoost regressor using TimeSeriesSplit cross-validation."""
    if df is None:
        df = generate_time_series_dataset()

    feature_cols = [
        "throughput_mbps",
        "packet_drop_percentage",
        "buffer_utilization_percentage",
        "node_temperature_celsius",
        "dynamic_operational_label",
        "lag_latency_1",
        "lag_latency_2",
        "throughput_slope",
        "buffer_peak"
    ]

    X = df[feature_cols]
    y = df["network_latency_ms"]

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=80,
        max_depth=4,
        learning_rate=0.07,
        random_state=42
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        cv_scores.append({"fold": fold + 1, "rmse": float(rmse), "mae": float(mae), "r2": float(r2)})

    # Final fit on complete historical data
    model.fit(X, y)
    save_path = output_path or settings.model_weights_path
    model.save_model(save_path)

    metrics = {
        "cv_scores": cv_scores,
        "mean_rmse": float(np.mean([s["rmse"] for s in cv_scores])),
        "mean_mae": float(np.mean([s["mae"] for s in cv_scores])),
        "mean_r2": float(np.mean([s["r2"] for s in cv_scores])),
        "model_file": save_path
    }

    logger.info("training_complete", mean_rmse=metrics["mean_rmse"], mean_r2=metrics["mean_r2"])
    return metrics


if __name__ == "__main__":
    train_time_series_model()
