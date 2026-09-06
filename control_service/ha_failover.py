"""
High-Availability Failover Coordinator.
Coordinates routing failover to standby PLC nodes (Node Delta) during link degradation or packet drop bursts.
"""
from typing import Dict, Any
from datetime import datetime, timezone
import structlog
from monitoring.metrics import HA_FAILOVER_EVENTS_TOTAL

logger = structlog.get_logger("helm-ha-failover")


class HAFailoverCoordinator:
    """Manages virtual route redirect and hot standby node activation."""

    def __init__(self, primary_node: str = "PLC_NODE_ALPHA", standby_node: str = "PLC_NODE_DELTA"):
        self.primary_node = primary_node
        self.standby_node = standby_node
        self.active_route = primary_node
        self.is_failover_active = False
        self.last_switch_time: datetime = datetime.now(timezone.utc)

    def trigger_failover(self, reason: str, from_node: str = None, to_node: str = None) -> Dict[str, Any]:
        """Switches traffic to the standby node."""
        source = from_node or self.primary_node
        target = to_node or self.standby_node

        self.active_route = target
        self.is_failover_active = True
        self.last_switch_time = datetime.now(timezone.utc)

        HA_FAILOVER_EVENTS_TOTAL.labels(from_node=source, to_node=target).inc()
        logger.warn("ha_failover_triggered", from_node=source, to_node=target, reason=reason)

        return {
            "status": "failover_engaged",
            "active_node": target,
            "standby_mode": "active_forwarding",
            "switched_at": self.last_switch_time.isoformat(),
            "reason": reason
        }

    def restore_primary(self) -> Dict[str, Any]:
        """Restores traffic routing back to primary hardware node."""
        self.active_route = self.primary_node
        self.is_failover_active = False
        self.last_switch_time = datetime.now(timezone.utc)

        logger.info("ha_primary_restored", active_node=self.primary_node)
        return {
            "status": "primary_restored",
            "active_node": self.primary_node,
            "switched_at": self.last_switch_time.isoformat()
        }


ha_coordinator = HAFailoverCoordinator()
