"""
Real-Time High-Precision Industrial Latency Probe (Phase 0).
Measures actual TCP socket handshake and ICMP round-trip time (RTT)
to physical endpoints, replacing synthetic formulas with ground truth.
"""
import socket
import time
import math
import statistics
import structlog
from typing import List, Dict, Any, Optional

logger = structlog.get_logger("helm-latency-probe")


class LatencyProbe:
    """
    Measures actual TCP/ICMP round-trip time to industrial endpoints.
    Replaces synthetic `network_latency_ms` generator with physical measurements.
    """

    def __init__(self, targets: Optional[List[Dict[str, Any]]] = None, protocol: str = "tcp", timeout_sec: float = 1.0):
        """
        targets: [{"host": "127.0.0.1", "port": 8003, "id": "ml-service"}, ...]
        """
        self.targets = targets or [{"host": "127.0.0.1", "port": 8003, "id": "default"}]
        self.protocol = protocol
        self.timeout_sec = timeout_sec

    def measure_tcp_rtt(self, host: str, port: int, samples: int = 5) -> float:
        """Measure TCP connection handshake latency in milliseconds."""
        rtts = []
        for _ in range(samples):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout_sec)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                t0 = time.perf_counter_ns()
                sock.connect((host, port))
                # Send minimal null probe byte
                sock.sendall(b"\x00")
                t1 = time.perf_counter_ns()
                sock.close()
                
                rtt_ms = (t1 - t0) / 1_000_000.0
                rtts.append(rtt_ms)
            except Exception:
                rtts.append(float("nan"))
            time.sleep(0.02)

        valid = [r for r in rtts if not math.isnan(r)]
        return float(statistics.median(valid)) if valid else 0.0

    def measure_icmp_ping(self, host: str, samples: int = 5) -> float:
        """Measures ICMP ping latency with fallback to TCP if raw sockets lack privileges."""
        try:
            from pythonping import ping
            response = ping(host, count=samples, timeout=self.timeout_sec)
            return float(response.rtt_avg_ms)
        except Exception as e:
            logger.warn("icmp_ping_fallback_to_tcp", host=host, reason=str(e))
            return self.measure_tcp_rtt(host, 80, samples=samples)

    def get_latency_for_target(self, target_id: str = "default") -> float:
        """Retrieves measured RTT for a specified target endpoint."""
        target = next((t for t in self.targets if t.get("id") == target_id), self.targets[0])
        host = target.get("host", "127.0.0.1")
        port = int(target.get("port", 8003))

        if self.protocol == "tcp":
            return self.measure_tcp_rtt(host, port)
        return self.measure_icmp_ping(host)


if __name__ == "__main__":
    probe = LatencyProbe(targets=[{"host": "127.0.0.1", "port": 8505, "id": "dashboard"}])
    print(f"Measured TCP Latency to Dashboard: {probe.get_latency_for_target('dashboard'):.3f} ms")
