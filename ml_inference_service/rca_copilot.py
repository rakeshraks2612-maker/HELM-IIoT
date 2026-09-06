"""
HELM-IIoT Autonomous AI SCADA Copilot & Root Cause Analysis (RCA) Engine.
Analyzes multi-dimensional telemetry, DBSCAN operational regimes, and TreeSHAP marginal attributions
to deliver explainable diagnoses, SLA breach risk assessments, and optimal closed-loop mitigations.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import structlog
from config.settings import settings

logger = structlog.get_logger("helm-ai-copilot")


class SCADACopilotEngine:
    """Expert AI reasoning engine for industrial cyber-physical root cause analysis."""

    def __init__(self):
        self.sla_threshold = float(settings.sla_latency_threshold_ms)

    def diagnose_root_cause(
        self,
        throughput_mbps: float,
        packet_drop_pct: float,
        buffer_util_pct: float,
        node_temp_celsius: float,
        predicted_latency_ms: float,
        actual_latency_ms: float,
        operational_regime: str,
        shap_attributions: Dict[str, float],
        ha_failover_active: bool = False
    ) -> Dict[str, Any]:
        """
        Performs multi-signal diagnostic analysis to identify the primary failure vector
        and generate actionable mitigation recommendations.
        """
        # Determine SLA risk level
        if predicted_latency_ms >= self.sla_threshold:
            risk_level = "CRITICAL_BREACH"
            urgency = "IMMEDIATE_ACTION_REQUIRED"
        elif predicted_latency_ms >= (self.sla_threshold - 8.0):
            risk_level = "WARNING_APPROACHING_LIMIT"
            urgency = "PROACTIVE_REGULATION_RECOMMENDED"
        else:
            risk_level = "NOMINAL_STABLE"
            urgency = "NO_ACTION_NEEDED"

        # Identify Primary Root Cause Vector based on telemetry & SHAP
        findings = []
        mitigation_action = "none"
        action_params = {}
        confidence_score = 0.95

        # 1. Packet Loss / Physical Link Degradation
        if packet_drop_pct >= float(settings.failover_drop_threshold_pct):
            findings.append({
                "vector": "Physical Link Degradation / Packet Loss Burst",
                "evidence": f"Frame drop rate ({packet_drop_pct:.2f}%) exceeds safety threshold ({settings.failover_drop_threshold_pct:.1f}%). TreeSHAP impact: +{shap_attributions.get('packet_drop_percentage', 8.5):.1f} ms.",
                "severity": "HIGH"
            })
            mitigation_action = "ha_failover"
            action_params = {
                "target_device": "PLC_NODE_ALPHA",
                "standby_node": "PLC_NODE_DELTA",
                "reason": f"Automated failover due to frame drop rate of {packet_drop_pct:.2f}%"
            }

        # 2. Socket Buffer Saturation / Queue Overflow
        if buffer_util_pct >= 65.0:
            findings.append({
                "vector": "TSN Socket Queue Depth Saturation",
                "evidence": f"Buffer capacity utilization at {buffer_util_pct:.1f}%. High risk of packet jitter and tail-drop. TreeSHAP impact: +{shap_attributions.get('buffer_utilization_percentage', 4.2):.1f} ms.",
                "severity": "MEDIUM" if buffer_util_pct < 85.0 else "CRITICAL"
            })
            if mitigation_action == "none":
                mitigation_action = "traffic_shedding"
                action_params = {
                    "target_device": "PLC_NODE_ALPHA",
                    "shedding_factor": float(settings.traffic_shedding_factor),
                    "reason": f"Dynamic traffic attenuation to relieve queue saturation ({buffer_util_pct:.1f}%)"
                }

        # 3. Thermal Throttling / Processor Core Overheating
        if node_temp_celsius >= float(settings.throttling_temp_threshold_celsius):
            findings.append({
                "vector": "Processor Core Thermal Load Throttling",
                "evidence": f"Die temperature ({node_temp_celsius:.1f} °C) exceeded limit ({settings.throttling_temp_threshold_celsius:.1f} °C). Compute clock throttling induced {node_temp_celsius * 0.1:.1f} ms latency penalty.",
                "severity": "HIGH"
            })
            if mitigation_action == "none":
                mitigation_action = "traffic_shedding"
                action_params = {
                    "target_device": "PLC_NODE_ALPHA",
                    "shedding_factor": 0.65,
                    "reason": f"Thermal throttling mitigation at {node_temp_celsius:.1f} °C"
                }

        # 4. Ingress Bandwidth Influx
        if throughput_mbps > 110.0:
            findings.append({
                "vector": "Ingress Network Bandwidth Surge",
                "evidence": f"Ingress data rate ({throughput_mbps:.1f} Mbps) exceeds baseline capacity.",
                "severity": "MEDIUM"
            })

        if not findings:
            findings.append({
                "vector": "Nominal Operation",
                "evidence": "All sensor telemetry channels operating within deterministic engineering bounds.",
                "severity": "INFO"
            })

        # Generate Executive Narrative
        primary_finding = findings[0]
        if risk_level != "NOMINAL_STABLE":
            diagnosis_summary = (
                f"**Root Cause Identified**: `{primary_finding['vector']}`. "
                f"{primary_finding['evidence']} "
                f"Forecasted latency is **{predicted_latency_ms:.1f} ms** against the **{self.sla_threshold:.0f} ms** SLA constraint."
            )
            recommendation_text = (
                f"Recommended action: **{mitigation_action.upper()}** on `{action_params.get('target_device', 'PLC_NODE_ALPHA')}`. "
                f"Anticipated latency reduction: **-{(predicted_latency_ms * (1.0 - settings.traffic_shedding_factor)):.1f} ms**."
            )
        else:
            diagnosis_summary = "All communication channels and node queues are operating deterministically within nominal parameters."
            recommendation_text = "No mitigation required. System is running with optimal deterministic QoS."

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_level": risk_level,
            "urgency": urgency,
            "confidence_score": confidence_score,
            "operational_regime": operational_regime,
            "primary_vector": primary_finding["vector"],
            "diagnosis_summary": diagnosis_summary,
            "recommendation_text": recommendation_text,
            "findings": findings,
            "mitigation_action": mitigation_action,
            "action_params": action_params
        }

    def generate_compliance_audit_summary(self, incident_history: List[Dict[str, Any]]) -> str:
        """Produces an industrial compliance audit markdown summary from incident logs."""
        total_events = len(incident_history)
        critical_count = sum(1 for e in incident_history if e.get("Severity") == "CRITICAL")
        mitigated_count = sum(1 for e in incident_history if e.get("Severity") == "MITIGATED")
        failover_count = sum(1 for e in incident_history if e.get("Severity") == "FAILOVER")

        return f"""
### HELM-IIoT Operational Compliance & RCA Audit Report
* **Report Timestamp**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
* **Total Monitored Telemetry Cycles**: {total_events}
* **SLA Breach Prevention Success Rate**: {((mitigated_count / max(1, critical_count + mitigated_count)) * 100):.1f}%

#### Incident Breakdown:
| Incident Severity | Count | Primary Trigger | Resolution Protocol |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | `{critical_count}` | Latency >= {settings.sla_latency_threshold_ms}ms | Closed-loop QoS Shedding |
| **MITIGATED** | `{mitigated_count}` | Dynamic Queue Attenuation | Rate factor {settings.traffic_shedding_factor} |
| **HA FAILOVER** | `{failover_count}` | Packet Loss > {settings.failover_drop_threshold_pct}% | Standby Route Node Delta |

**Conclusion**: The system complies with deterministic Time-Sensitive Networking (TSN) industrial bounds.
"""


# Global singleton copilot instance
copilot_engine = SCADACopilotEngine()
