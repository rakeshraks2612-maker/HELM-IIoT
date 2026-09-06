from scada_ui.client import HelmServicesClient


def test_system_end_to_end_client():
    client = HelmServicesClient()

    # 1. Test Physical RTT Probing
    rtt = client.measure_physical_rtt()
    assert isinstance(rtt, float)
    assert rtt > 0.0

    # 2. Test ML Latency Prediction & Regime Categorization
    pred_res = client.predict_latency(
        throughput=55.0,
        drop_pct=0.2,
        buffer_util=35.0,
        temp=42.0,
        slope=0.5,
        buffer_peak=40.0
    )
    assert "predicted_latency_ms" in pred_res
    assert "dynamic_operational_label" in pred_res
    assert "cluster_regime" in pred_res
    assert pred_res["predicted_latency_ms"] > 0.0

    # 3. Test Closed-Loop Mitigation Execution
    mitig_res = client.execute_mitigation(
        action="traffic_shedding",
        target_device="PLC_NODE_ALPHA",
        reason="Predicted SLA latency threshold violation"
    )
    assert mitig_res["status"] in ["success", "rate_limited"]

    # 4. Test Reset
    reset_res = client.execute_mitigation(
        action="reset",
        target_device="PLC_NODE_ALPHA",
        reason="Return to nominal state"
    )
    assert reset_res["status"] in ["success", "rate_limited"]
