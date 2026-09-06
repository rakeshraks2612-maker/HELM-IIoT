"""
Control & QoS Mitigation Microservice.
Provides secure REST endpoints for traffic shedding, thermal management, and failover routing.
"""
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Depends, Security, Response
from fastapi.security.api_key import APIKeyHeader
from config.settings import settings
from edge_gateway.schemas import MitigationCommand
from control_service.traffic_shaper import traffic_shaper
from control_service.ha_failover import ha_coordinator
from control_service.safety_guard import safety_guard
from monitoring.logger import logger
from monitoring.metrics import MITIGATION_ACTIONS_TOTAL, get_latest_metrics

app = FastAPI(
    title="HELM-IIoT Control & Mitigation Service",
    version="2.0.0",
    description="Deterministic QoS enforcement, Linux TC rate shaping, and HA standby switchover"
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
        "service": "control-mitigation",
        "dead_mans_switch_tripped": safety_guard.is_dead_mans_switch_tripped(),
        "ha_active_node": ha_coordinator.active_route,
        "active_tc_limits": traffic_shaper.active_limits
    }


@app.get("/metrics")
async def metrics():
    body, content_type = get_latest_metrics()
    return Response(content=body, media_type=content_type)


@app.post("/mitigate/execute")
async def execute_mitigation(cmd: MitigationCommand, _auth=Depends(verify_api_key)):
    """Executes closed-loop QoS mitigation with anti-flapping debounce verification."""
    # Check rate limiter
    can_proceed, wait_time = safety_guard.check_rate_limit()
    if not can_proceed:
        logger.warn("mitigation_rate_limited", wait_seconds=wait_time, correlation_id=cmd.correlation_id)
        return {
            "status": "rate_limited",
            "message": f"Action throttled. Please wait {wait_time}s to avoid flapping.",
            "retry_after_sec": wait_time
        }

    # Human-in-the-loop permission verification
    if cmd.requires_approval and not cmd.approved_by:
        logger.warn("mitigation_requires_approval", action=cmd.action)
        return {
            "status": "pending_approval",
            "message": "Action requires manual engineering approval before applying.",
            "action": cmd.action
        }

    result = {}
    if cmd.action == "traffic_shedding":
        target_rate = cmd.target_rate_mbps or (100.0 * (cmd.shedding_factor or settings.traffic_shedding_factor))
        result = traffic_shaper.apply_traffic_shaping(rate_mbps=target_rate, interface=cmd.target_interface)
        MITIGATION_ACTIONS_TOTAL.labels(action_type="traffic_shedding", severity="WARNING").inc()

    elif cmd.action == "ha_failover":
        result = ha_coordinator.trigger_failover(
            reason=cmd.reason,
            from_node=cmd.target_device,
            to_node=cmd.standby_node
        )
        MITIGATION_ACTIONS_TOTAL.labels(action_type="ha_failover", severity="CRITICAL").inc()

    elif cmd.action == "reset":
        shaper_res = traffic_shaper.remove_traffic_shaping(interface=cmd.target_interface)
        ha_res = ha_coordinator.restore_primary()
        result = {"traffic_shaping": shaper_res, "ha_route": ha_res, "status": "nominal_restored"}
        MITIGATION_ACTIONS_TOTAL.labels(action_type="reset", severity="INFO").inc()

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mitigation action: {cmd.action}")

    safety_guard.record_mitigation_action()
    logger.info("mitigation_executed", action=cmd.action, result=result, correlation_id=cmd.correlation_id)

    return {
        "status": "success",
        "action": cmd.action,
        "details": result,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "correlation_id": cmd.correlation_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.control_service_host, port=settings.control_service_port)
