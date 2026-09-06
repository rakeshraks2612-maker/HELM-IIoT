from ml_inference_service.rca_copilot import SCADACopilotEngine


def test_copilot_nominal_state():
    engine = SCADACopilotEngine()
    diag = engine.diagnose_root_cause(
        throughput_mbps=55.0,
        packet_drop_pct=0.2,
        buffer_util_pct=35.0,
        node_temp_celsius=42.0,
        predicted_latency_ms=41.5,
        actual_latency_ms=41.2,
        operational_regime="Nominal Regime",
        shap_attributions={"packet_drop_percentage": 0.5, "buffer_utilization_percentage": 0.2}
    )
    assert diag["risk_level"] == "NOMINAL_STABLE"
    assert diag["mitigation_action"] == "none"
    assert diag["confidence_score"] >= 0.90


def test_copilot_packet_loss_failover():
    engine = SCADACopilotEngine()
    diag = engine.diagnose_root_cause(
        throughput_mbps=45.0,
        packet_drop_pct=3.8,  # > 1.8% threshold
        buffer_util_pct=50.0,
        node_temp_celsius=44.0,
        predicted_latency_ms=88.5,
        actual_latency_ms=85.0,
        operational_regime="Anomaly Vector",
        shap_attributions={"packet_drop_percentage": 14.5}
    )
    assert diag["risk_level"] == "CRITICAL_BREACH"
    assert diag["mitigation_action"] == "ha_failover"
    assert diag["action_params"]["standby_node"] == "PLC_NODE_DELTA"


def test_copilot_buffer_saturation_traffic_shedding():
    engine = SCADACopilotEngine()
    diag = engine.diagnose_root_cause(
        throughput_mbps=78.0,
        packet_drop_pct=0.5,
        buffer_util_pct=88.0,  # Saturation
        node_temp_celsius=45.0,
        predicted_latency_ms=64.0,
        actual_latency_ms=62.0,
        operational_regime="Congestion State",
        shap_attributions={"buffer_utilization_percentage": 8.5}
    )
    assert diag["risk_level"] == "CRITICAL_BREACH"
    assert diag["mitigation_action"] == "traffic_shedding"
    assert diag["action_params"]["shedding_factor"] > 0.0


def test_copilot_compliance_report():
    engine = SCADACopilotEngine()
    incidents = [
        {"Timestamp": "10:00:00", "Severity": "INFO", "Event": "System nominal"},
        {"Timestamp": "10:05:00", "Severity": "CRITICAL", "Event": "SLA breach"},
        {"Timestamp": "10:05:05", "Severity": "MITIGATED", "Event": "QoS shedding active"},
        {"Timestamp": "10:10:00", "Severity": "FAILOVER", "Event": "Node Delta standby switch"}
    ]
    report = engine.generate_compliance_audit_summary(incidents)
    assert "HELM-IIoT Operational Compliance" in report
    assert "CRITICAL" in report
    assert "MITIGATED" in report
