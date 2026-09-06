"""
Automated unit and integration test suite for Hardware-in-the-Loop (HIL) Bridge.
"""
import pytest
from edge_gateway.hil_connector import HILHardwareBridge, hil_bridge
from scada_ui.client import helm_client


def test_hil_bridge_initialization():
    bridge = HILHardwareBridge(mode="virtual_s7", target_ip="192.168.10.14", target_port=102)
    assert bridge.mode == "virtual_s7"
    assert bridge.target_ip == "192.168.10.14"
    assert bridge.target_port == 102
    assert not bridge.is_connected


def test_hil_virtual_connection():
    bridge = HILHardwareBridge(mode="virtual_s7")
    status = bridge.connect()
    assert status["is_connected"] is True
    assert status["frames_tx"] >= 1
    assert status["frames_rx"] >= 1


def test_hil_hardware_telemetry_polling():
    bridge = HILHardwareBridge(mode="virtual_s7")
    bridge.connect()
    data = bridge.poll_hardware()
    assert data["connected"] is True
    assert "telemetry" in data
    telem = data["telemetry"]
    assert 10.0 <= telem["throughput_mbps"] <= 160.0
    assert 0.0 <= telem["packet_drop_pct"] <= 10.0
    assert 30.0 <= telem["node_temperature_c"] <= 90.0
    assert "gpio_port_mask" in telem
    assert data["rtt_ms"] > 0.0


def test_hil_frame_buffer_ring():
    bridge = HILHardwareBridge(mode="virtual_s7")
    bridge.connect()
    for _ in range(5):
        bridge.poll_hardware()
    frames = bridge.frame_buffer
    assert len(frames) > 0
    assert "hex" in frames[-1]
    assert "type" in frames[-1]


def test_hil_client_sdk_integration():
    res_conn = helm_client.connect_hil(mode="virtual_s7", target_ip="192.168.10.14", port=102)
    assert res_conn["is_connected"] is True
    
    poll_res = helm_client.poll_hil_hardware()
    assert poll_res["connected"] is True
    
    status_res = helm_client.get_hil_status()
    assert status_res["is_connected"] is True
    
    frames = helm_client.get_hil_frames()
    assert isinstance(frames, list)
    
    disc_res = helm_client.disconnect_hil()
    assert disc_res["is_connected"] is False
