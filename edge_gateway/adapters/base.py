"""
Base Protocol Adapter Interface for Industrial Fieldbus Ingestion.
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any
import structlog
from edge_gateway.schemas import RawSensorPayload

logger = structlog.get_logger("helm-protocol-adapter")


class BaseProtocolAdapter(ABC):
    """Abstract base class for non-blocking edge protocol adapters."""

    def __init__(self, name: str, device_id: str):
        self.name = name
        self.device_id = device_id
        self.is_connected = False
        self.ingest_callback: Optional[Callable[[RawSensorPayload], None]] = None

    def register_callback(self, callback: Callable[[RawSensorPayload], None]):
        """Registers listener callback for decoded canonical sensor frames."""
        self.ingest_callback = callback

    @abstractmethod
    async def connect(self) -> bool:
        """Establishes connection to the fieldbus broker/PLC."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Cleanly tears down protocol connection."""
        pass

    @abstractmethod
    async def poll_or_listen(self):
        """Asynchronous execution loop for ingesting frames."""
        pass

    def emit_payload(self, payload: RawSensorPayload):
        """Dispatches normalized payload to the registered pipeline callback."""
        if self.ingest_callback:
            try:
                self.ingest_callback(payload)
            except Exception as e:
                logger.error("adapter_dispatch_failed", adapter=self.name, error=str(e))
