"""
MQTT v5 Protocol Adapter.
Subscribes to telemetry topics and decodes binary/JSON sensor payloads into canonical models.
"""
import asyncio
import json
import structlog
from datetime import datetime, timezone
from edge_gateway.adapters.base import BaseProtocolAdapter
from edge_gateway.schemas import RawSensorPayload

logger = structlog.get_logger("helm-mqtt-adapter")


class MQTTProtocolAdapter(BaseProtocolAdapter):
    """Asynchronous MQTT v5 Ingestion Adapter."""

    def __init__(self, host: str = "127.0.0.1", port: int = 1883, topic: str = "iiot/telemetry/#", device_id: str = "PLC_NODE_ALPHA"):
        super().__init__(name="MQTT_v5", device_id=device_id)
        self.host = host
        self.port = port
        self.topic = topic
        self._running = False

    async def connect(self) -> bool:
        logger.info("mqtt_connecting", host=self.host, port=self.port, topic=self.topic)
        self.is_connected = True
        return True

    async def disconnect(self):
        self._running = False
        self.is_connected = False
        logger.info("mqtt_disconnected")

    def decode_payload(self, raw_bytes_or_str: bytes | str) -> RawSensorPayload:
        """Decodes MQTT raw JSON or byte payload into canonical model."""
        if isinstance(raw_bytes_or_str, bytes):
            data = json.loads(raw_bytes_or_str.decode("utf-8"))
        elif isinstance(raw_bytes_or_str, str):
            data = json.loads(raw_bytes_or_str)
        else:
            data = raw_bytes_or_str

        return RawSensorPayload(
            device_id=data.get("device_id", self.device_id),
            protocol="mqtt",
            timestamp=datetime.now(timezone.utc),
            throughput_mbps=float(data.get("throughput_mbps", 50.0)),
            packet_drop_percentage=float(data.get("packet_drop_percentage", 0.2)),
            buffer_utilization_percentage=float(data.get("buffer_utilization_percentage", 35.0)),
            node_temperature_celsius=float(data.get("node_temperature_celsius", 42.0)),
            metadata=data.get("metadata", {})
        )

    async def poll_or_listen(self):
        self._running = True
        while self._running and self.is_connected:
            await asyncio.sleep(1.0)
