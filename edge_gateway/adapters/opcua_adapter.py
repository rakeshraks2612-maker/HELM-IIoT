"""
OPC-UA Protocol Adapter.
Subscribes to OPC-UA server node telemetry values for deterministic automation systems.
"""
import asyncio
import structlog
from datetime import datetime, timezone
from edge_gateway.adapters.base import BaseProtocolAdapter
from edge_gateway.schemas import RawSensorPayload

logger = structlog.get_logger("helm-opcua-adapter")


class OPCUAProtocolAdapter(BaseProtocolAdapter):
    """OPC-UA Ingestion Adapter."""

    def __init__(self, endpoint_url: str = "opc.tcp://127.0.0.1:4840/freeopcua/server/", device_id: str = "PLC_NODE_GAMMA"):
        super().__init__(name="OPC_UA", device_id=device_id)
        self.endpoint_url = endpoint_url
        self._running = False

    async def connect(self) -> bool:
        logger.info("opcua_connecting", endpoint=self.endpoint_url)
        self.is_connected = True
        return True

    async def disconnect(self):
        self._running = False
        self.is_connected = False
        logger.info("opcua_disconnected")

    def decode_node_dict(self, node_values: dict) -> RawSensorPayload:
        """Parses OPC-UA node value map into canonical schema."""
        return RawSensorPayload(
            device_id=self.device_id,
            protocol="opcua",
            timestamp=datetime.now(timezone.utc),
            throughput_mbps=float(node_values.get("ns=2;s=Throughput", 55.0)),
            packet_drop_percentage=float(node_values.get("ns=2;s=PacketLoss", 0.15)),
            buffer_utilization_percentage=float(node_values.get("ns=2;s=BufferUtil", 45.0)),
            node_temperature_celsius=float(node_values.get("ns=2;s=CoreTemp", 43.5)),
            metadata={"endpoint": self.endpoint_url}
        )

    async def poll_or_listen(self):
        self._running = True
        while self._running and self.is_connected:
            await asyncio.sleep(1.0)
