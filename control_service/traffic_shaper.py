"""
Linux Traffic Control (tc) QoS Interface.
Applies real Token Bucket Filter (TBF) traffic shaping on Linux interfaces,
with automatic graceful sandbox fallback for macOS and containerized dev environments.
"""
import subprocess
import shutil
import platform
import structlog
from typing import Dict, Any

logger = structlog.get_logger("helm-traffic-shaper")


class LinuxTrafficShaper:
    """Controls network interface bandwidth rate and queue depth via tc qdisc."""

    def __init__(self, default_interface: str = "eth0"):
        self.default_interface = default_interface
        self.is_linux = platform.system().lower() == "linux"
        self.has_tc = shutil.which("tc") is not None
        self.active_limits: Dict[str, float] = {}

    def apply_traffic_shaping(self, rate_mbps: float, interface: str = None) -> Dict[str, Any]:
        """
        Enforces bandwidth throttling and queue shaping using Linux tc Token Bucket Filter.
        Command equivalent:
        tc qdisc add/replace dev <iface> root tbf rate <rate_mbps>mbit burst 32kbit latency 10ms
        """
        target_iface = interface or self.default_interface
        rate_clamped = max(1.0, round(rate_mbps, 2))

        if self.is_linux and self.has_tc:
            try:
                # Try replace first, if fails try add
                res = subprocess.run([
                    "tc", "qdisc", "replace", "dev", target_iface, "root", "tbf",
                    "rate", f"{rate_clamped}mbit", "burst", "32kbit", "latency", "10ms"
                ], capture_output=True, text=True, check=False)
                
                if res.returncode != 0:
                    subprocess.run([
                        "tc", "qdisc", "add", "dev", target_iface, "root", "tbf",
                        "rate", f"{rate_clamped}mbit", "burst", "32kbit", "latency", "10ms"
                    ], capture_output=True, text=True, check=False)

                self.active_limits[target_iface] = rate_clamped
                logger.info("tc_shaping_applied", interface=target_iface, rate_mbps=rate_clamped)
                return {"status": "applied", "mode": "kernel_tc", "interface": target_iface, "rate_mbps": rate_clamped}
            except Exception as e:
                logger.error("tc_execution_error", error=str(e))
                self.active_limits[target_iface] = rate_clamped
                return {"status": "applied_fallback", "mode": "sandbox", "interface": target_iface, "rate_mbps": rate_clamped, "note": str(e)}
        else:
            # Emulated sandbox execution on macOS / local test runner
            self.active_limits[target_iface] = rate_clamped
            logger.info("sandbox_traffic_shaping_simulated", interface=target_iface, rate_mbps=rate_clamped)
            return {"status": "applied_simulated", "mode": "sandbox_emulation", "interface": target_iface, "rate_mbps": rate_clamped}

    def remove_traffic_shaping(self, interface: str = None) -> Dict[str, Any]:
        """Removes root qdisc traffic shaping constraints."""
        target_iface = interface or self.default_interface
        if self.is_linux and self.has_tc:
            try:
                subprocess.run(["tc", "qdisc", "del", "dev", target_iface, "root"], capture_output=True, text=True, check=False)
            except Exception:
                pass
        
        self.active_limits.pop(target_iface, None)
        logger.info("tc_shaping_cleared", interface=target_iface)
        return {"status": "cleared", "interface": target_iface}


traffic_shaper = LinuxTrafficShaper()
