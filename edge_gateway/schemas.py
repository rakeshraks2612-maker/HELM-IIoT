"""
Canonical Telemetry Schemas and Validation Models.
Enforces strict industrial bounds and data quality gates.
"""
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class RawSensorPayload(BaseModel):
    """Raw payload from industrial fieldbuses (MQTT, Modbus, OPC-UA)."""
    device_id: str = Field(..., description="Unique PLC identifier, e.g., PLC_NODE_ALPHA")
    protocol: str = Field(..., description="Source protocol (mqtt, modbus_tcp, opcua, coap)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    throughput_mbps: float = Field(..., ge=0.0, le=10000.0, description="Network throughput in Mbps")
    packet_drop_percentage: float = Field(..., ge=0.0, le=100.0, description="Packet loss rate (%)")
    buffer_utilization_percentage: float = Field(..., ge=0.0, le=100.0, description="Socket/queue buffer usage (%)")
    node_temperature_celsius: float = Field(..., ge=-20.0, le=125.0, description="Core processor thermal load")
    actual_rtt_latency_ms: Optional[float] = Field(default=None, ge=0.0, le=5000.0, description="True measured RTT")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("device_id cannot be empty")
        return v.strip().upper()


class CanonicalTelemetryFrame(BaseModel):
    """Standardized normalized frame for streaming ingestion and feature store."""
    device_id: str
    protocol: str
    timestamp_epoch_ms: int
    throughput_mbps: float
    packet_drop_percentage: float
    buffer_utilization_percentage: float
    node_temperature_celsius: float
    actual_rtt_latency_ms: float
    correlation_id: str


class PredictionRequest(BaseModel):
    """Inference request payload for supervised XGBoost / ONNX model."""
    throughput_mbps: float = Field(..., ge=0.0)
    packet_drop_percentage: float = Field(..., ge=0.0, le=100.0)
    buffer_utilization_percentage: float = Field(..., ge=0.0, le=100.0)
    node_temperature_celsius: float = Field(..., ge=-20.0, le=120.0)
    dynamic_operational_label: int = Field(default=0)
    throughput_slope: Optional[float] = Field(default=0.0)
    buffer_peak: Optional[float] = Field(default=None)
    lag_latency_1: Optional[float] = Field(default=None)
    lag_latency_2: Optional[float] = Field(default=None)


class PredictionResponse(BaseModel):
    """Inference response."""
    predicted_latency_ms: float
    sla_violation_expected: bool
    dynamic_operational_label: int
    cluster_regime: str
    inference_duration_ms: float
    model_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MitigationCommand(BaseModel):
    """Control plane action command."""
    action: str = Field(..., description="traffic_shedding, thermal_regulation, ha_failover, reset")
    target_device: str
    target_interface: str = "eth0"
    shedding_factor: Optional[float] = None
    target_rate_mbps: Optional[float] = None
    standby_node: Optional[str] = "PLC_NODE_DELTA"
    reason: str
    requires_approval: bool = False
    approved_by: Optional[str] = None
    correlation_id: str
