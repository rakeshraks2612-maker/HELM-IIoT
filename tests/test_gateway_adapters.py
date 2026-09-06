import pytest
import asyncio
from edge_gateway.schemas import RawSensorPayload
from edge_gateway.adapters.mqtt_adapter import MQTTProtocolAdapter
from edge_gateway.adapters.modbus_adapter import ModbusTCPAdapter
from edge_gateway.adapters.opcua_adapter import OPCUAProtocolAdapter


def test_mqtt_payload_decoder():
    adapter = MQTTProtocolAdapter()
    raw_json = '{"device_id": "PLC_NODE_ALPHA", "throughput_mbps": 88.5, "packet_drop_percentage": 0.45, "buffer_utilization_percentage": 50.0, "node_temperature_celsius": 45.0}'
    payload = adapter.decode_payload(raw_json)
    assert payload.device_id == "PLC_NODE_ALPHA"
    assert payload.throughput_mbps == 88.5
    assert payload.packet_drop_percentage == 0.45
    assert payload.protocol == "mqtt"


def test_modbus_register_decoder():
    adapter = ModbusTCPAdapter(device_id="PLC_NODE_BETA")
    # Regs: [Throughput*10, Drops*100, Buffer*10, Temp*10]
    regs = [620, 35, 450, 480]
    payload = adapter.decode_registers(regs)
    assert payload.device_id == "PLC_NODE_BETA"
    assert payload.throughput_mbps == 62.0
    assert payload.packet_drop_percentage == 0.35
    assert payload.buffer_utilization_percentage == 45.0
    assert payload.node_temperature_celsius == 48.0


def test_opcua_node_decoder():
    adapter = OPCUAProtocolAdapter(device_id="PLC_NODE_GAMMA")
    node_map = {
        "ns=2;s=Throughput": 74.2,
        "ns=2;s=PacketLoss": 0.12,
        "ns=2;s=BufferUtil": 38.0,
        "ns=2;s=CoreTemp": 41.5
    }
    payload = adapter.decode_node_dict(node_map)
    assert payload.device_id == "PLC_NODE_GAMMA"
    assert payload.throughput_mbps == 74.2
    assert payload.packet_drop_percentage == 0.12
