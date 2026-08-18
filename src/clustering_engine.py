# =====================================================================
# MODULE 2: UNSUPERVISED DENSITY CLUSTERING & LABELLING
# =====================================================================
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

def execute_autonomous_profiling(csv_path="processed_telemetry.csv"):
    """Clusters chaotic IoT telemetry streams to assign dynamic state labels."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[Error] '{csv_path}' not found. Please run data_pipeline.py first.")
        return None
        
    feature_cols = ["throughput_mbps", "packet_drop_percentage", "buffer_utilization_percentage", "node_temperature_celsius"]
    matrix = df[feature_cols].values
    
    print("[Phase 3] Computing density profiles via DBSCAN framework...")
    # EPS maps localized cluster radius, Min_samples handles outliers/noise bounds
    dbscan = DBSCAN(eps=0.30, min_samples=8)
    labels = dbscan.fit_predict(matrix)
    
    # Check if clustering layout successfully isolated state distributions
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        print("DBSCAN density uniform. Falling back to structured K-Means array mapping...")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)
        
    df["dynamic_operational_label"] = labels
    
    # Mathematical Silhouette Coefficient Verification Loop
    if len(set(labels)) > 1:
        sil_coef = silhouette_score(matrix, labels)
        print(f"[Verification Metric] Silhouette Analytics Validation Score: {sil_coef:.4f}")
    else:
        print("[Warning] Single density layer mapped. Model parameter refinement required.")
        
    return df

if __name__ == "__main__":
    profiled_df = execute_autonomous_profiling()
    if profiled_df is not None:
        profiled_df.to_csv("labeled_telemetry.csv", index=False)
        print("Unsupervised state profiling sequence completed. Output mapped successfully.")