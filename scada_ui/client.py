"""
HELM Microservices Client SDK for SCADA HUD.
Encapsulates communication with Edge Gateway, Feature Store, ML Inference, and Control services.
"""
import httpx
from typing import Dict, Any, Optional, List
from config.settings import settings
from edge_gateway.schemas import RawSensorPayload, PredictionRequest, MitigationCommand
from edge_gateway.latency_prober import SocketLatencyProber
from ml_inference_service.server import model_container
from ml_inference_service.online_cluster import online_profiler
from control_service.traffic_shaper import traffic_shaper
from control_service.ha_failover import ha_coordinator
from control_service.safety_guard import safety_guard


class HelmServicesClient:
    """Client for querying microservices or running direct fallback engine."""

    def __init__(self):
        self.api_key = settings.api_key
        self.headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        self.prober = SocketLatencyProber(
            target_host=settings.target_probe_host,
            target_port=settings.target_probe_port
        )

    def measure_physical_rtt(self) -> float:
        """Measures true physical socket RTT."""
        rtt, ok = self.prober.measure_tcp_rtt()
        return rtt

    def predict_latency(
        self,
        throughput: float,
        drop_pct: float,
        buffer_util: float,
        temp: float,
        slope: float = 0.0,
        buffer_peak: float = 0.0,
        lag_1: float = 40.0,
        lag_2: float = 40.0
    ) -> Dict[str, Any]:
        """Predicts end-to-end response latency and classifies operational regime."""
        cluster_id, regime_name = online_profiler.predict_regime(
            throughput=throughput,
            drop_pct=drop_pct,
            buffer_util=buffer_util,
            temp=temp
        )

        req = PredictionRequest(
            throughput_mbps=throughput,
            packet_drop_percentage=drop_pct,
            buffer_utilization_percentage=buffer_util,
            node_temperature_celsius=temp,
            dynamic_operational_label=cluster_id,
            throughput_slope=slope,
            buffer_peak=buffer_peak or buffer_util,
            lag_latency_1=lag_1,
            lag_latency_2=lag_2
        )

        predicted_ms = model_container.predict(req)
        is_sla_violated = predicted_ms >= settings.sla_latency_threshold_ms

        return {
            "predicted_latency_ms": predicted_ms,
            "sla_violation_expected": is_sla_violated,
            "dynamic_operational_label": cluster_id,
            "cluster_regime": regime_name,
            "model_version": model_container.model_version
        }

    def execute_mitigation(
        self,
        action: str,
        target_device: str,
        reason: str,
        shedding_factor: float = None,
        correlation_id: str = "ui-client"
    ) -> Dict[str, Any]:
        """Executes QoS traffic shaping, HA failover, or reset."""
        can_proceed, wait_sec = safety_guard.check_rate_limit()
        if not can_proceed:
            return {
                "status": "rate_limited",
                "message": f"Rate limited. Wait {wait_sec}s.",
                "wait_sec": wait_sec
            }

        result = {}
        if action == "traffic_shedding":
            factor = shedding_factor or settings.traffic_shedding_factor
            rate = 100.0 * factor
            result = traffic_shaper.apply_traffic_shaping(rate_mbps=rate)
        elif action == "ha_failover":
            result = ha_coordinator.trigger_failover(reason=reason, from_node=target_device)
        elif action == "reset":
            shaper_res = traffic_shaper.remove_traffic_shaping()
            ha_res = ha_coordinator.restore_primary()
            result = {"traffic": shaper_res, "ha": ha_res}

        safety_guard.record_mitigation_action()
        return {"status": "success", "action": action, "result": result}


# Global client instance
helm_client = HelmServicesClient()
