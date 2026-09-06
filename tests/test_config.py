import pytest
from config.settings import HelmConfig, settings


def test_settings_defaults():
    assert settings.environment in ["development", "staging", "production"]
    assert settings.sla_latency_threshold_ms == 60.0
    assert settings.traffic_shedding_factor == 0.82
    assert settings.dbscan_eps == 0.30
    assert settings.api_key is not None


def test_settings_validation_bounds():
    with pytest.raises(Exception):
        HelmConfig(sla_latency_threshold_ms=1.0)  # Below min 5.0

    with pytest.raises(Exception):
        HelmConfig(traffic_shedding_factor=1.5)   # Above max 1.0
