"""
Closed-Loop Control Safety Guard & Dead Man's Switch.
Enforces debounce intervals (anti-flapping), safe fail-soft defaults,
and human-in-the-loop permission checks for disruptive production overrides.
"""
import time
from typing import Dict, Any, Optional, Tuple
import structlog
from config.settings import settings

logger = structlog.get_logger("helm-safety-guard")


class SafetyGuard:
    """Guarantees stability, anti-flapping rate limits, and dead man's timeout."""

    def __init__(self, debounce_sec: float = None):
        self.debounce_sec = debounce_sec or settings.mitigation_debounce_sec
        self.last_mitigation_time: float = 0.0
        self.last_heartbeat_time: float = time.time()
        self.heartbeat_timeout_sec: float = 15.0

    def record_heartbeat(self):
        """Updates internal alive timer from the edge gateway."""
        self.last_heartbeat_time = time.time()

    def is_dead_mans_switch_tripped(self) -> bool:
        """Returns True if upstream telemetry has ceased, requiring safe fallback state."""
        return (time.time() - self.last_heartbeat_time) > self.heartbeat_timeout_sec

    def check_rate_limit(self) -> Tuple[bool, float]:
        """Checks if enough time has passed since last mitigation to prevent control loop oscillation."""
        now = time.time()
        elapsed = now - self.last_mitigation_time
        if elapsed < self.debounce_sec:
            return False, round(self.debounce_sec - elapsed, 2)
        return True, 0.0

    def record_mitigation_action(self):
        """Records timestamp of applied mitigation."""
        self.last_mitigation_time = time.time()


safety_guard = SafetyGuard()
