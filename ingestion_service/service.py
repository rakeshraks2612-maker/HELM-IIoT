"""
Ingestion & Feature Store Microservice.
Provides endpoints for streaming buffer queries, feature extractions, and subscriber feeds.
"""
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Depends, Security, Response
from fastapi.security.api_key import APIKeyHeader
from config.settings import settings
from edge_gateway.schemas import CanonicalTelemetryFrame, PredictionRequest
from ingestion_service.bus import event_bus
from ingestion_service.feature_store import feature_store
from monitoring.metrics import get_latest_metrics

app = FastAPI(
    title="HELM-IIoT Ingestion & Feature Store Service",
    version="2.0.0",
    description="Real-time feature computation, lag windowing, and stream buffering"
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    if settings.enable_auth:
        if not api_key or api_key != settings.api_key:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing API key")
    return api_key


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "ingestion-feature-store",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/metrics")
async def metrics():
    body, content_type = get_latest_metrics()
    return Response(content=body, media_type=content_type)


@app.post("/stream/push")
async def push_telemetry_frame(frame: CanonicalTelemetryFrame, _auth=Depends(verify_api_key)):
    """Receives canonical frame from edge gateway and registers into feature store & event bus."""
    feature_store.push_frame(frame)
    await event_bus.publish(frame)
    return {"status": "enqueued", "device_id": frame.device_id, "correlation_id": frame.correlation_id}


@app.get("/features/{device_id}")
async def get_device_features(device_id: str, _auth=Depends(verify_api_key)):
    """Fetches real-time computed rolling features for a specific device."""
    feat = feature_store.get_features(device_id)
    if not feat:
        raise HTTPException(status_code=404, detail=f"No telemetry found for device: {device_id}")
    return feat


@app.get("/features/{device_id}/inference_payload", response_model=PredictionRequest)
async def get_inference_payload(device_id: str, dynamic_label: int = 0, _auth=Depends(verify_api_key)):
    """Returns preprocessed prediction request vector for downstream ML service."""
    req = feature_store.to_prediction_request(device_id, dynamic_label=dynamic_label)
    if not req:
        raise HTTPException(status_code=404, detail=f"Insufficient history for device: {device_id}")
    return req


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.ingestion_host, port=settings.ingestion_port)
