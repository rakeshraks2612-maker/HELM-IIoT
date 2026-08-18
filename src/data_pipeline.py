# =====================================================================
# MODULE 1: DATA INGESTION & FEATURE ENGINEERING
# =====================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_industrial_telemetry(n_samples=1200):
    """Simulates chaotic, raw, unlabelled multi-sensor network telemetry."""
    np.random.seed(42)
    time_ticks = pd.date_range(start="2026-07-14 09:00:00", periods=n_samples, freq="s")
    
    # Simulating 4 standard sensor traffic metrics
    throughput = np.random.normal(loc=50, scale=10, size=n_samples)
    packet_drop_rate = np.random.exponential(scale=2, size=n_samples)
    buffer_utilization = np.random.uniform(low=10, high=90, size=n_samples)
    node_temperature = np.random.normal(loc=42, scale=3, size=n_samples)
    
    # Injecting periodic volatile congestion states (chaotic noise logs)
    for i in range(100, 250):
        packet_drop_rate[i] += 12
        buffer_utilization[i] += 40
    for i in range(700, 850):
        throughput[i] -= 25
        packet_drop_rate[i] += 8

    df = pd.DataFrame({
        "timestamp": time_ticks,
        "throughput_mbps": throughput,
        "packet_drop_percentage": packet_drop_rate,
        "buffer_utilization_percentage": buffer_utilization,
        "node_temperature_celsius": node_temperature
    })
    return df

def process_and_scale_data(df):
    """Executes feature normalization and noise imputation."""
    feature_cols = ["throughput_mbps", "packet_drop_percentage", "buffer_utilization_percentage", "node_temperature_celsius"]
    
    # Missing value verification loop (Forward fill time-series data drops)
    df[feature_cols] = df[feature_cols].ffill()
    
    # Feature Scaling transformation array
    scaler = MinMaxScaler()
    scaled_matrix = scaler.fit_transform(df[feature_cols])
    
    scaled_df = pd.DataFrame(scaled_matrix, columns=feature_cols)
    scaled_df["timestamp"] = df["timestamp"]
    return scaled_df, scaler

if __name__ == "__main__":
    print("[Phase 2] Initializing data preprocessing module...")
    raw_data = generate_industrial_telemetry()
    scaled_df, scaler = process_and_scale_data(raw_data)
    scaled_df.to_csv("processed_telemetry.csv", index=False)
    print(f"Data scaled successfully. Matrix Shape: {scaled_df.shape}. Saved to storage disk.")