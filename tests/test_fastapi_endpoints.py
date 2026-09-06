import pytest
from fastapi.testclient import TestClient
from config.settings import settings
from edge_gateway.gateway_service import app as gateway_app
from ingestion_service.service import app as ingestion_app
from ml_inference_service.server import app as ml_app
from control_service.service import app as control_app


@pytest.fixture
def auth_headers():
    return {"X-API-Key": settings.api_key}


def test_gateway_endpoints(auth_headers):
    client = TestClient(gateway_app)
    
    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["service"] == "edge-gateway"

    # 2. Ingest raw frame with auth
    raw_payload = {
        "device_id": "PLC_NODE_ALPHA",
        "protocol": "mqtt",
        "throughput_mbps": 82.5,
        "packet_drop_percentage": 0.25,
        "buffer_utilization_percentage": 30.0,
        "node_temperature_celsius": 42.0
    }
    res_ingest = client.post("/ingest/raw", json=raw_payload, headers=auth_headers)
    assert res_ingest.status_code == 200
    data = res_ingest.json()
    assert data["device_id"] == "PLC_NODE_ALPHA"
    assert "actual_rtt_latency_ms" in data

    # 3. Ingest raw frame without auth should fail
    res_unauth = client.post("/ingest/raw", json=raw_payload)
    assert res_unauth.status_code == 403


def test_ingestion_endpoints(auth_headers):
    client = TestClient(ingestion_app)

    # Health
    res_health = client.get("/health")
    assert res_health.status_code == 200

    # Push canonical frame
    frame = {
        "device_id": "PLC_NODE_BETA",
        "protocol": "modbus_tcp",
        "timestamp_epoch_ms": 1700000000000,
        "throughput_mbps": 70.0,
        "packet_drop_percentage": 0.1,
        "buffer_utilization_percentage": 40.0,
        "node_temperature_celsius": 41.0,
        "actual_rtt_latency_ms": 39.5,
        "correlation_id": "test-uuid"
    }
    res_push = client.post("/stream/push", json=frame, headers=auth_headers)
    assert res_push.status_code == 200

    # Fetch features
    res_feat = client.get("/features/PLC_NODE_BETA", headers=auth_headers)
    assert res_feat.status_code == 200
    assert res_feat.json()["throughput_mbps"] == 70.0


def test_ml_service_endpoints(auth_headers):
    client = TestClient(ml_app)

    res_health = client.get("/health")
    assert res_health.status_code == 200

    # Predict
    pred_req = {
        "throughput_mbps": 55.0,
        "packet_drop_percentage": 0.2,
        "buffer_utilization_percentage": 35.0,
        "node_temperature_celsius": 42.0,
        "dynamic_operational_label": 0
    }
    res_pred = client.post("/predict", json=pred_req, headers=auth_headers)
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert "predicted_latency_ms" in data
    assert "cluster_regime" in data
    assert "shap_attributions" in data
    assert isinstance(data["shap_attributions"], dict)

    # Drift status
    res_drift = client.get("/drift/status", headers=auth_headers)
    assert res_drift.status_code == 200

    # Retrain with admin role
    res_retrain = client.post("/retrain", headers={"X-API-Key": settings.admin_api_key})
    assert res_retrain.status_code == 200


def test_control_service_endpoints(auth_headers):
    client = TestClient(control_app)

    res_health = client.get("/health")
    assert res_health.status_code == 200

    # Execute traffic shedding
    mitig_cmd = {
        "action": "traffic_shedding",
        "target_device": "PLC_NODE_ALPHA",
        "target_interface": "eth0",
        "shedding_factor": 0.82,
        "reason": "Predicted SLA breach",
        "correlation_id": "test-cmd-01"
    }
    res_mitig = client.post("/mitigate/execute", json=mitig_cmd, headers=auth_headers)
    assert res_mitig.status_code == 200
    assert res_mitig.json()["status"] in ["success", "rate_limited"]
