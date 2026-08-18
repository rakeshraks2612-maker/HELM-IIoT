import os
import sys
# pyrefly: ignore [missing-import]
import pytest
import pandas as pd
import numpy as np

# Add src to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# pyrefly: ignore [missing-import]
from data_pipeline import generate_industrial_telemetry, process_and_scale_data
# pyrefly: ignore [missing-import]
from clustering_engine import execute_autonomous_profiling
# pyrefly: ignore [missing-import]
from ensemble_training import train_predictive_engine

def test_generate_industrial_telemetry():
    df = generate_industrial_telemetry(n_samples=1200)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1200
    expected_cols = [
        "timestamp",
        "throughput_mbps",
        "packet_drop_percentage",
        "buffer_utilization_percentage",
        "node_temperature_celsius"
    ]
    for col in expected_cols:
        assert col in df.columns

def test_process_and_scale_data():
    df = generate_industrial_telemetry(n_samples=1200)
    # Inject a NaN to test forward-fill
    df.loc[10, "throughput_mbps"] = np.nan
    
    scaled_df, scaler = process_and_scale_data(df)
    assert not scaled_df["throughput_mbps"].isnull().any()
    
    # Assert values are scaled (mostly between 0 and 1)
    for col in ["throughput_mbps", "packet_drop_percentage", "buffer_utilization_percentage", "node_temperature_celsius"]:
        assert scaled_df[col].min() >= 0.0 - 1e-9
        assert scaled_df[col].max() <= 1.0 + 1e-9

def test_execute_autonomous_profiling(tmp_path):
    # Prepare dummy processed data
    df = generate_industrial_telemetry(n_samples=1200)
    scaled_df, _ = process_and_scale_data(df)
    
    csv_file = tmp_path / "processed_telemetry.csv"
    scaled_df.to_csv(csv_file, index=False)
    
    profiled_df = execute_autonomous_profiling(csv_path=str(csv_file))
    assert profiled_df is not None
    assert "dynamic_operational_label" in profiled_df.columns

def test_train_predictive_engine(tmp_path):
    # Prepare dummy labeled data
    df = generate_industrial_telemetry(n_samples=1200)
    scaled_df, _ = process_and_scale_data(df)
    scaled_df["dynamic_operational_label"] = np.random.choice([0, 1, 2], size=len(scaled_df))
    
    csv_file = tmp_path / "labeled_telemetry.csv"
    scaled_df.to_csv(csv_file, index=False)
    
    # Save current CWD and change to tmp_path to prevent overwriting workspace engine
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        train_predictive_engine(csv_path=str(csv_file))
        assert os.path.exists("predictive_edge_engine.json")
    finally:
        os.chdir(old_cwd)
