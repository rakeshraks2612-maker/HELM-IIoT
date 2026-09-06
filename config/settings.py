"""
HELM-IIoT Configuration Management.
Centralized, validated settings using Pydantic BaseSettings and environment variables.
"""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HelmConfig(BaseSettings):
    """Production configuration for HELM-IIoT cyber-physical edge stack."""
    model_config = SettingsConfigDict(
        env_prefix="HELM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Environment
    environment: str = Field(default="development", description="development, staging, production")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Service Bindings & Ports
    gateway_host: str = Field(default="0.0.0.0")
    gateway_port: int = Field(default=8001)
    
    ingestion_host: str = Field(default="0.0.0.0")
    ingestion_port: int = Field(default=8002)

    ml_service_host: str = Field(default="0.0.0.0")
    ml_service_port: int = Field(default=8003)

    control_service_host: str = Field(default="0.0.0.0")
    control_service_port: int = Field(default=8004)

    scada_ui_port: int = Field(default=8505)
    metrics_port: int = Field(default=9090)

    # Security & Auth & RBAC
    api_key: str = Field(default="helm-sec-key-industrial-2026", description="Internal service API key")
    operator_api_key: str = Field(default="helm-operator-key-2026", description="Operator role API key")
    admin_api_key: str = Field(default="helm-admin-key-2026", description="Admin/Engineer role API key")
    jwt_secret: str = Field(default="helm-jwt-secret-deterministic-tsn-994", description="JWT secret key")
    enable_auth: bool = Field(default=True)

    # SLA & Control Thresholds
    sla_latency_threshold_ms: float = Field(default=60.0, ge=5.0, le=500.0, description="Target max latency SLA limit")
    traffic_shedding_factor: float = Field(default=0.82, ge=0.1, le=1.0, description="Rate multiplier during congestion")
    throttling_temp_threshold_celsius: float = Field(default=68.0, ge=30.0, le=100.0)
    failover_drop_threshold_pct: float = Field(default=1.8, ge=0.1, le=20.0)
    mitigation_debounce_sec: float = Field(default=5.0, ge=1.0, le=60.0, description="Anti-flapping safety window")

    # ML, ONNX & Clustering Parameters
    dbscan_eps: float = Field(default=0.30, ge=0.01, le=2.0)
    dbscan_min_samples: int = Field(default=8, ge=2, le=50)
    model_weights_path: str = Field(default="predictive_edge_engine.json")
    onnx_model_path: str = Field(default="predictive_edge_engine.onnx")
    drift_psi_threshold: float = Field(default=0.20, ge=0.05, le=1.0)
    auto_retrain_on_drift: bool = Field(default=True)
    
    # Feature Store & Windows
    feature_window_sec: int = Field(default=30, ge=5, le=300)
    rolling_buffer_capacity: int = Field(default=1000, ge=100)

    # Protocol & Hardware Prober
    network_interface: str = Field(default="eth0")
    target_probe_host: str = Field(default="127.0.0.1")
    target_probe_port: int = Field(default=8003)
    mqtt_broker_host: str = Field(default="127.0.0.1")
    mqtt_broker_port: int = Field(default=1883)
    modbus_server_host: str = Field(default="127.0.0.1")
    modbus_server_port: int = Field(default=5020)
    opcua_server_url: str = Field(default="opc.tcp://127.0.0.1:4840/freeopcua/server/")


# Global singleton settings instance
settings = HelmConfig()
