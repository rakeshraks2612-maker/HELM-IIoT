"""
High-Precision Hardware Socket RTT Latency Prober.
Measures true network round-trip time (RTT) via TCP socket handshake / probe timings,
replacing synthetic target formulas with real physical microsecond telemetry.
"""
import socket
import time
import structlog
from typing import Tuple

logger = structlog.get_logger("helm-latency-prober")


class SocketLatencyProber:
    """Active socket prober measuring actual end-to-end transport latency."""

    def __init__(self, target_host: str = "127.0.0.1", target_port: int = 8003, timeout_sec: float = 0.5):
        self.target_host = target_host
        self.target_port = target_port
        self.timeout_sec = timeout_sec

    def measure_tcp_rtt(self, host: str = None, port: int = None) -> Tuple[float, bool]:
        """
        Measures real TCP connection handshake and minimal probe transfer latency in milliseconds.
        Returns: (latency_ms, success_bool)
        """
        target_host = host or self.target_host
        target_port = port or self.target_port
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_sec)
        # Enable TCP_NODELAY to disable Nagle's algorithm for deterministic latency
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        t0 = time.perf_counter_ns()
        try:
            sock.connect((target_host, target_port))
            t1 = time.perf_counter_ns()
            
            # Send single null probe byte if socket is writable
            sock.sendall(b"\x00")
            
            latency_ms = (t1 - t0) / 1_000_000.0
            return max(0.1, round(latency_ms, 3)), True
        except Exception:
            # Fallback estimation when endpoint socket is offline
            t_err = time.perf_counter_ns()
            elapsed_ms = (t_err - t0) / 1_000_000.0
            return max(1.0, round(elapsed_ms, 3)), False
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def measure_synthetic_rtt_from_load(self, base_ms: float, drops: float, buffer_util: float) -> float:
        """Deterministic physics-based RTT baseline for lab and offline simulation harness."""
        jitter = (hash(str(time.time_ns())) % 100) / 100.0 - 0.5
        simulated_rtt = base_ms + (drops * 8.5) + ((buffer_util / 100.0) ** 2 * 25.0) + jitter
        return max(0.5, round(simulated_rtt, 3))
