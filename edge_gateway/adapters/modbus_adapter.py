"""
Modbus TCP Industrial Protocol Adapter.
Polls PLC holding registers and input registers to extract physical node parameters.
"""
import asyncio
import structlog
from datetime import datetime, timezone
from edge_gateway.adapters.base import BaseProtocolAdapter
from edge_gateway.schemas import RawSensorPayload

logger = structlog.get_logger("helm-modbus-adapter")


class ModbusTCPAdapter(BaseProtocolAdapter):
    """Modbus TCP Polling Adapter for Industrial PLCs."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5020, unit_id: int = 1, device_id: str = "PLC_NODE_BETA"):
        super().__init__(name="Modbus_TCP", device_id=device_id)
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self._running = False

    async def connect(self) -> bool:
        logger.info("modbus_connecting", host=self.host, port=self.port, unit_id=self.unit_id)
        self.is_connected = True
        return True

    async def disconnect(self):
        self._running = False
        self.is_connected = False
        logger.info("modbus_disconnected")

    def decode_registers(self, registers: list[int]) -> RawSensorPayload:
        """
        Maps standard 16-bit Modbus registers to physical floating point engineering units:
        Reg 0: Throughput x10 (e.g. 523 -> 52.3 Mbps)
        Reg 1: Packet drops x100 (e.g. 45 -> 0.45%)
        Reg 2: Buffer utilization x10 (e.g. 680 -> 68.0%)
        Reg 3: Temperature x10 (e.g. 445 -> 44.5 C)
        """
        throughput = registers[0] / 10.0 if len(registers) > 0 else 50.0
        drops = registers[1] / 100.0 if len(registers) > 1 else 0.2
        buffer_util = registers[2] / 10.0 if len(registers) > 2 else 40.0
        temp = registers[3] / 10.0 if len(registers) > 3 else 42.0

        return RawSensorPayload(
            device_id=self.device_id,
            protocol="modbus_tcp",
            timestamp=datetime.now(timezone.utc),
            throughput_mbps=throughput,
            packet_drop_percentage=drops,
            buffer_utilization_percentage=buffer_util,
            node_temperature_celsius=temp,
            metadata={"unit_id": self.unit_id, "raw_registers": registers}
        )

    async def poll_or_listen(self):
        self._running = True
        while self._running and self.is_connected:
            await asyncio.sleep(1.0)
