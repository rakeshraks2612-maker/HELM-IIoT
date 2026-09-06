"""
Prometheus Metrics Exporter for HELM-IIoT cyber-physical telemetry and inference tracking.
"""
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Ingress & Protocol Metrics
INGRESS_FRAMES_TOTAL = Counter(
    "helm_ingress_frames_total",
    "Total raw sensor frames ingested across protocols",
    ["protocol", "device_id"]
)

INGRESS_ERRORS_TOTAL = Counter(
    "helm_ingress_errors_total",
    "Total ingestion or decoding failures",
    ["protocol", "error_type"]
)

# True Hardware Latency Metrics
ACTUAL_RTT_LATENCY_MS = Gauge(
    "helm_actual_rtt_latency_ms",
    "Real measured socket round-trip time in milliseconds",
    ["target_host"]
)

# Predictive Model Metrics
PREDICTED_LATENCY_MS = Gauge(
    "helm_predicted_latency_ms",
    "Forecasted end-to-end network latency in milliseconds",
    ["model_version"]
)

INFERENCE_LATENCY_HISTOGRAM = Histogram(
    "helm_inference_duration_seconds",
    "Time taken to execute XGBoost edge tree inference",
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05]
)

DATA_DRIFT_PSI = Gauge(
    "helm_data_drift_psi",
    "Current Population Stability Index measuring telemetry drift",
    ["feature_name"]
)

# Mitigation & Control Metrics
MITIGATION_ACTIONS_TOTAL = Counter(
    "helm_mitigation_actions_total",
    "Number of closed-loop QoS mitigations triggered",
    ["action_type", "severity"]
)

HA_FAILOVER_EVENTS_TOTAL = Counter(
    "helm_ha_failover_events_total",
    "High availability standby route switchovers",
    ["from_node", "to_node"]
)

ACTIVE_QOS_BANDWIDTH_LIMIT_MBPS = Gauge(
    "helm_active_qos_bandwidth_limit_mbps",
    "Currently enforced Linux TC bandwidth rate in Mbps"
)


def get_latest_metrics() -> tuple[bytes, str]:
    """Returns current Prometheus metric scrapings and content type header."""
    return generate_latest(), CONTENT_TYPE_LATEST
