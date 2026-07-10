from importlib import metadata

import logfire
import pytest

from haiku.rag import telemetry


@pytest.fixture
def captured_configure(monkeypatch):
    """Capture the kwargs telemetry.configure() passes to logfire.configure,
    and no-op the instrumentation so tests don't touch a real exporter."""
    captured: dict = {}

    def _fake_configure(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(logfire, "configure", _fake_configure)
    monkeypatch.setattr(logfire, "instrument_pydantic_ai", lambda: None)
    return captured


def test_default_service_name_used_when_env_unset(captured_configure, monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("LOGFIRE_SERVICE_NAME", raising=False)

    telemetry.configure(service_name="haiku-ingester")

    assert captured_configure["service_name"] == "haiku-ingester"


def test_otel_service_name_overrides_default(captured_configure, monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "customer-ingester")

    telemetry.configure(service_name="haiku-ingester")

    # Deferring to logfire (service_name=None) lets it read the env var,
    # so the customer's OTEL_SERVICE_NAME wins over our default.
    assert captured_configure["service_name"] is None


def test_logfire_service_name_overrides_default(captured_configure, monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.setenv("LOGFIRE_SERVICE_NAME", "customer-ingester")

    telemetry.configure(service_name="haiku-ingester")

    assert captured_configure["service_name"] is None


def test_service_version_is_package_version(captured_configure, monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("LOGFIRE_SERVICE_NAME", raising=False)

    telemetry.configure(service_name="haiku-rag")

    assert captured_configure["service_version"] == metadata.version("haiku.rag-slim")
