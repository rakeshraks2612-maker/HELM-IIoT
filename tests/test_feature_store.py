from edge_gateway.schemas import CanonicalTelemetryFrame
from ingestion_service.feature_store import RealTimeFeatureStore


def test_feature_store_rolling_window():
    store = RealTimeFeatureStore(window_capacity=10)
    device = "PLC_NODE_ALPHA"

    for i in range(5):
        frame = CanonicalTelemetryFrame(
            device_id=device,
            protocol="mqtt",
            timestamp_epoch_ms=1000 + i * 1000,
            throughput_mbps=50.0 + i * 5.0,
            packet_drop_percentage=0.2 + i * 0.1,
            buffer_utilization_percentage=30.0 + i * 2.0,
            node_temperature_celsius=40.0,
            actual_rtt_latency_ms=41.0 + i * 1.5,
            correlation_id=f"corr-{i}"
        )
        store.push_frame(frame)

    features = store.get_features(device)
    assert features is not None
    assert features["samples_in_window"] == 5
    assert features["throughput_slope"] > 0  # Monotonically increasing throughput
    assert features["buffer_peak"] == 38.0
    assert features["lag_latency_1"] == 45.5  # Previous frame actual latency


def test_to_prediction_request():
    store = RealTimeFeatureStore()
    device = "PLC_NODE_BETA"
    frame = CanonicalTelemetryFrame(
        device_id=device,
        protocol="modbus_tcp",
        timestamp_epoch_ms=2000,
        throughput_mbps=60.0,
        packet_drop_percentage=0.5,
        buffer_utilization_percentage=45.0,
        node_temperature_celsius=42.0,
        actual_rtt_latency_ms=43.0,
        correlation_id="corr-99"
    )
    store.push_frame(frame)
    req = store.to_prediction_request(device, dynamic_label=1)
    assert req is not None
    assert req.throughput_mbps == 60.0
    assert req.dynamic_operational_label == 1
