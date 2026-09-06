from edge_gateway.latency_prober import SocketLatencyProber


def test_socket_latency_prober_fallback():
    prober = SocketLatencyProber(target_host="127.0.0.1", target_port=9999, timeout_sec=0.1)
    rtt, success = prober.measure_tcp_rtt()
    assert isinstance(rtt, float)
    assert rtt > 0.0


def test_synthetic_rtt_model():
    prober = SocketLatencyProber()
    rtt_nominal = prober.measure_synthetic_rtt_from_load(base_ms=38.0, drops=0.1, buffer_util=20.0)
    rtt_congested = prober.measure_synthetic_rtt_from_load(base_ms=38.0, drops=3.5, buffer_util=90.0)
    assert rtt_congested > rtt_nominal
