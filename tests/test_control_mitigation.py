import time
from control_service.traffic_shaper import LinuxTrafficShaper
from control_service.ha_failover import HAFailoverCoordinator
from control_service.safety_guard import SafetyGuard


def test_traffic_shaping():
    shaper = LinuxTrafficShaper()
    res = shaper.apply_traffic_shaping(rate_mbps=82.0, interface="test0")
    assert res["status"] in ["applied", "applied_simulated", "applied_fallback"]
    assert "test0" in shaper.active_limits

    clear_res = shaper.remove_traffic_shaping(interface="test0")
    assert clear_res["status"] == "cleared"


def test_ha_failover():
    coordinator = HAFailoverCoordinator(primary_node="NODE_A", standby_node="NODE_D")
    assert coordinator.active_route == "NODE_A"

    failover_res = coordinator.trigger_failover(reason="Packet loss > 2%")
    assert coordinator.is_failover_active is True
    assert coordinator.active_route == "NODE_D"

    restore_res = coordinator.restore_primary()
    assert coordinator.is_failover_active is False
    assert coordinator.active_route == "NODE_A"


def test_safety_guard_debounce():
    guard = SafetyGuard(debounce_sec=0.2)
    can_proceed_1, _ = guard.check_rate_limit()
    assert can_proceed_1 is True

    guard.record_mitigation_action()
    can_proceed_2, wait = guard.check_rate_limit()
    assert can_proceed_2 is False
    assert wait > 0.0

    time.sleep(0.25)
    can_proceed_3, _ = guard.check_rate_limit()
    assert can_proceed_3 is True
