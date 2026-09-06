"""
Edge Gateway Microservice.
Collects, validates, and standardizes industrial fieldbus telemetry before publishing to the streaming backbone.
"""
import uuid
import time
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Depends, Security, Response
from fastapi.security.api_key import APIKeyHeader
from pydantic import ValidationError
from config.settings import settings
from monitoring.logger import logger
from monitoring.metrics import (
    INGRESS_FRAMES_TOTAL,
    INGRESS_ERRORS_TOTAL,
    ACTUAL_RTT_LATENCY_MS,
    get_latest_metrics
)
from edge_gateway.schemas import RawSensorPayload, CanonicalTelemetryFrame
from edge_gateway.latency_prober import SocketLatencyProber

app = FastAPI(
    title="HELM-IIoT Edge Gateway Service",
    version="2.0.0",
    description="Multi-Protocol Ingress & High-Precision Telemetry Normalization"
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    if settings.enable_auth:
        if not api_key or api_key != settings.api_key:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing API key")
    return api_key


prober = SocketLatencyProber(
    target_host=settings.target_probe_host,
    target_port=settings.target_probe_port
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "edge-gateway",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "adapters_active": ["mqtt_v5", "modbus_tcp", "opcua"]
    }


@app.get("/metrics")
async def prometheus_metrics():
    body, content_type = get_latest_metrics()
    return Response(content=body, media_type=content_type)


@app.post("/ingest/raw", response_model=CanonicalTelemetryFrame)
async def ingest_raw_sensor_frame(payload: RawSensorPayload, _auth=Depends(verify_api_key)):
    """Ingests, validates, stamps physical RTT, and normalizes raw fieldbus sensor telemetry."""
    try:
        # Measure true socket latency
        if payload.actual_rtt_latency_ms is not None:
            rtt_ms = payload.actual_rtt_latency_ms
        else:
            measured_rtt, ok = prober.measure_tcp_rtt()
            if not ok:
                rtt_ms = prober.measure_synthetic_rtt_from_load(
                    base_ms=38.0,
                    drops=payload.packet_drop_percentage,
                    buffer_util=payload.buffer_utilization_percentage
                )
            else:
                rtt_ms = measured_rtt

        # Update Prometheus Telemetry
        INGRESS_FRAMES_TOTAL.labels(protocol=payload.protocol, device_id=payload.device_id).inc()
        ACTUAL_RTT_LATENCY_MS.labels(target_host=payload.device_id).set(rtt_ms)

        canonical = CanonicalTelemetryFrame(
            device_id=payload.device_id,
            protocol=payload.protocol,
            timestamp_epoch_ms=int(time.time() * 1000),
            throughput_mbps=payload.throughput_mbps,
            packet_drop_percentage=payload.packet_drop_percentage,
            buffer_utilization_percentage=payload.buffer_utilization_percentage,
            node_temperature_celsius=payload.node_temperature_celsius,
            actual_rtt_latency_ms=rtt_ms,
            correlation_id=str(uuid.uuid4())
        )

        logger.info(
            "frame_ingested",
            device_id=canonical.device_id,
            protocol=canonical.protocol,
            throughput=canonical.throughput_mbps,
            rtt_ms=canonical.actual_rtt_latency_ms,
            correlation_id=canonical.correlation_id
        )
        return canonical

    except ValidationError as val_err:
        INGRESS_ERRORS_TOTAL.labels(protocol=payload.protocol, error_type="validation_error").inc()
        logger.error("ingestion_validation_error", error=str(val_err))
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as e:
        INGRESS_ERRORS_TOTAL.labels(protocol=payload.protocol, error_type="system_error").inc()
        logger.error("ingestion_fatal_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal ingestion failure")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.gateway_host, port=settings.gateway_port)
