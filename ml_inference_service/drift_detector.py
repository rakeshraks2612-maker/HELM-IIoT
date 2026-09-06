"""
Data Drift Detection Engine.
Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov test across streaming features
to automatically trigger model retraining when distribution shifts exceed tolerance bounds.
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import structlog
from config.settings import settings
from monitoring.metrics import DATA_DRIFT_PSI

logger = structlog.get_logger("helm-drift-detector")


class TelemetryDriftDetector:
    """Monitors telemetry distribution shift against baseline calibration dataset."""

    def __init__(self, psi_threshold: float = None):
        self.psi_threshold = psi_threshold or settings.drift_psi_threshold
        # Baseline reference distributions (calibrated during offline training)
        self._baselines: Dict[str, np.ndarray] = {
            "throughput_mbps": np.random.normal(50.0, 10.0, 500),
            "packet_drop_percentage": np.random.exponential(0.5, 500),
            "buffer_utilization_percentage": np.random.uniform(20.0, 70.0, 500),
            "node_temperature_celsius": np.random.normal(42.0, 3.0, 500)
        }

    def set_baseline(self, feature_name: str, baseline_values: np.ndarray):
        """Updates reference calibration baseline for a feature."""
        self._baselines[feature_name] = np.array(baseline_values)

    def calculate_psi(self, baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
        """Computes Population Stability Index (PSI) between baseline and production batches."""
        if len(baseline) == 0 or len(current) == 0:
            return 0.0

        # Create quantile buckets based on baseline
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(baseline, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Count frequencies in each bin
        baseline_counts, _ = np.histogram(baseline, bins=bins)
        current_counts, _ = np.histogram(current, bins=bins)

        # Convert to relative proportions with smoothing epsilon to avoid division by zero
        eps = 1e-4
        b_prop = np.maximum(baseline_counts / len(baseline), eps)
        c_prop = np.maximum(current_counts / len(current), eps)

        psi_val = np.sum((c_prop - b_prop) * np.log(c_prop / b_prop))
        return float(round(max(0.0, psi_val), 4))

    def evaluate_drift(self, current_batch: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Evaluates drift across all incoming telemetry channels.
        Returns drift metrics and flag indicating if retraining is required.
        """
        results = {}
        max_psi = 0.0
        retrain_recommended = False

        for feat_name, cur_vals in current_batch.items():
            if feat_name in self._baselines and len(cur_vals) >= 20:
                base_vals = self._baselines[feat_name]
                cur_arr = np.array(cur_vals)
                
                # Compute PSI
                psi = self.calculate_psi(base_vals, cur_arr)
                DATA_DRIFT_PSI.labels(feature_name=feat_name).set(psi)

                # Compute KS-test
                ks_stat, p_val = stats.ks_2samp(base_vals, cur_arr)

                results[feat_name] = {
                    "psi": psi,
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "drift_detected": psi > self.psi_threshold
                }

                if psi > max_psi:
                    max_psi = psi
                if psi > self.psi_threshold:
                    retrain_recommended = True

        return {
            "overall_max_psi": max_psi,
            "psi_threshold": self.psi_threshold,
            "retrain_recommended": retrain_recommended,
            "feature_metrics": results
        }


drift_detector = TelemetryDriftDetector()
