# =====================================================================
# MODULE 3: SUPERVISED GRADIENT-BOOSTED ENSEMBLE FORECASTING
# =====================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_predictive_engine(csv_path="labeled_telemetry.csv"):
    """Trains a high-velocity tree-boosting engine on dynamically labeled matrices."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[Error] '{csv_path}' not found. Please run clustering_engine.py first.")
        return
    
    # Simulating the physical response domain variable: Network Latency (ms)
    # Latency scaling driven by raw drop percentages and dynamic anomaly cluster profiles
    base_latency = 40 + (df["packet_drop_percentage"] * 35) + (df["dynamic_operational_label"].abs() * 12)
    df["network_latency_ms"] = base_latency + np.random.normal(0, 2, size=len(df))
    
    # Constructing temporal and physics features for the ML model pipeline
    df["lag_latency_1"] = df["network_latency_ms"].shift(1).bfill()
    df["lag_latency_2"] = df["network_latency_ms"].shift(2).bfill()
    df["throughput_slope"] = df["throughput_mbps"].diff().fillna(0.0)
    df["buffer_peak"] = df["buffer_utilization_percentage"].rolling(10, min_periods=1).max()

    features = [
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
    X = df[features]
    y = df["network_latency_ms"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[Phase 4] Initiating low-overhead ensemble training framework...")
    # Constraining max_depth and estimators ensures edge-processing friendliness on M4 Air
    predictive_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=80,
        max_depth=4,
        learning_rate=0.07,
        random_state=42
    )
    
    predictive_model.fit(X_train, y_train)
    
    # Root-Mean-Square Error (RMSE) evaluation verification
    predictions = predictive_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"[Verification Metric] Tree Model Prediction Matrix Validation RMSE: {rmse:.4f} ms")
    
    # Save the finalized pipeline state weights to disk
    predictive_model.save_model("predictive_edge_engine.json")
    print("Gradient-boosted predictive model compiled and written to execution memory.")

if __name__ == "__main__":
    train_predictive_engine()