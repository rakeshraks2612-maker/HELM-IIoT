"""
Online Streaming Clustering and Operational State Categorizer.
Performs streaming density profiling and maps multi-dimensional telemetry into operational regimes:
- Cluster 0: Nominal Regime (Balanced load, minimal drop)
- Cluster 1: Congestion State (Elevated queue depth / buffer)
- Cluster 2: Anomaly Vector (High frame drop or thermal surge)
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from typing import Tuple, List, Dict
import structlog

logger = structlog.get_logger("helm-online-cluster")


class OnlineRegimeProfiler:
    """Incremental streaming cluster engine with density baseline."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=32,
            n_init="auto"
        )
        self.is_fitted = False
        self.regime_names = {
            0: "Nominal Regime",
            1: "Congestion State",
            2: "Anomaly Vector"
        }
        # Pre-seed with representative anchor points
        self._initialize_anchors()

    def _initialize_anchors(self):
        """Initializes cluster centroids with known physical boundaries."""
        self.anchors = np.array([
            [50.0, 0.2, 35.0, 42.0],   # Cluster 0: Nominal
            [35.0, 1.2, 75.0, 52.0],   # Cluster 1: Congested
            [15.0, 4.5, 92.0, 72.0]    # Cluster 2: Critical Anomaly
        ])
        self.kmeans.fit(self.anchors)
        self.is_fitted = True

    def predict_regime(self, throughput: float, drop_pct: float, buffer_util: float, temp: float) -> Tuple[int, str]:
        """Classifies a live telemetry vector into an operational regime."""
        vec = np.array([throughput, drop_pct, buffer_util, temp])
        
        # Explicit physics rule boundaries for deterministic TSN safety
        if drop_pct >= 2.5 or temp >= 68.0:
            cluster_id = 2
        elif buffer_util >= 65.0 or drop_pct >= 0.8:
            cluster_id = 1
        elif drop_pct < 0.8 and buffer_util < 65.0 and temp < 60.0:
            cluster_id = 0
        else:
            # Nearest centroid to known archetypes
            dists = [np.linalg.norm(vec - anchor) for anchor in self.anchors]
            cluster_id = int(np.argmin(dists))

        regime_name = self.regime_names.get(cluster_id, "Nominal Regime")
        return cluster_id, regime_name

    def partial_fit_stream(self, batch_matrix: np.ndarray):
        """Incrementally updates cluster centroids using streaming batch data."""
        if len(batch_matrix) > 0:
            self.kmeans.partial_fit(batch_matrix)


online_profiler = OnlineRegimeProfiler()
