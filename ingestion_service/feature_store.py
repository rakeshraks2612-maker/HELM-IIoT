"""
Real-Time Feature Store and Temporal Feature Engine.
Computes rolling window statistics, lag features, and first-order rate derivatives (slopes).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import deque
from edge_gateway.schemas import CanonicalTelemetryFrame, PredictionRequest
import structlog

logger = structlog.get_logger("helm-feature-store")


class RealTimeFeatureStore:
    """Computes streaming temporal features and prevents training-serving skew."""

    def __init__(self, window_capacity: int = 60):
        self.window_capacity = window_capacity
        # Per-device rolling frame history
        self._device_buffers: Dict[str, deque] = {}

    def push_frame(self, frame: CanonicalTelemetryFrame):
        """Appends incoming telemetry frame into the device window."""
        device_id = frame.device_id
        if device_id not in self._device_buffers:
            self._device_buffers[device_id] = deque(maxlen=self.window_capacity)
        self._device_buffers[device_id].append(frame)

    def get_features(self, device_id: str) -> Optional[Dict[str, float]]:
        """
        Computes real-time feature dictionary for a given device:
        - Rolling means & peaks
        - First-order throughput trend slope
        - Lag metrics (latency at t-1, t-2)
        """
        buf = self._device_buffers.get(device_id)
        if not buf or len(buf) == 0:
            return None

        recent = list(buf)
        throughputs = [f.throughput_mbps for f in recent]
        drops = [f.packet_drop_percentage for f in recent]
        buffers = [f.buffer_utilization_percentage for f in recent]
        temps = [f.node_temperature_celsius for f in recent]
        latencies = [f.actual_rtt_latency_ms for f in recent]

        # Calculate slope via polynomial fit
        n = len(throughputs)
        if n >= 3:
            slope = float(np.polyfit(range(n), throughputs, 1)[0])
        else:
            slope = 0.0

        # Extract lag features
        lag_1 = latencies[-2] if n >= 2 else latencies[-1]
        lag_2 = latencies[-3] if n >= 3 else lag_1

        features = {
            "throughput_mbps": float(throughputs[-1]),
            "throughput_mean": float(np.mean(throughputs)),
            "throughput_slope": float(round(slope, 4)),
            "packet_drop_percentage": float(drops[-1]),
            "packet_drop_mean": float(np.mean(drops)),
            "buffer_utilization_percentage": float(buffers[-1]),
            "buffer_peak": float(np.max(buffers)),
            "node_temperature_celsius": float(temps[-1]),
            "actual_rtt_latency_ms": float(latencies[-1]),
            "lag_latency_1": float(lag_1),
            "lag_latency_2": float(lag_2),
            "samples_in_window": n
        }
        return features

    def to_prediction_request(self, device_id: str, dynamic_label: int = 0) -> Optional[PredictionRequest]:
        """Formats feature store output directly into an ML inference payload."""
        feat = self.get_features(device_id)
        if not feat:
            return None

        return PredictionRequest(
            throughput_mbps=feat["throughput_mbps"],
            packet_drop_percentage=feat["packet_drop_percentage"],
            buffer_utilization_percentage=feat["buffer_utilization_percentage"],
            node_temperature_celsius=feat["node_temperature_celsius"],
            dynamic_operational_label=dynamic_label,
            throughput_slope=feat["throughput_slope"],
            buffer_peak=feat["buffer_peak"],
            lag_latency_1=feat["lag_latency_1"],
            lag_latency_2=feat["lag_latency_2"]
        )


# Global feature store singleton
feature_store = RealTimeFeatureStore()
