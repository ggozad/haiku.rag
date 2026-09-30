import asyncio
from importlib import metadata

import logfire
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

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


def test_scrubbing_defaults_to_enabled(captured_configure):
    telemetry.configure(service_name="haiku-rag")

    # None is logfire's "scrubbing enabled" default.
    assert captured_configure["scrubbing"] is None


def test_scrubbing_can_be_disabled(captured_configure):
    telemetry.configure(service_name="evals", scrubbing=False)

    assert captured_configure["scrubbing"] is False


# --- Joining a parent trace passed in TRACEPARENT ---
#
# The autouse `isolate_env_trace_context` fixture (tests/conftest.py) clears
# TRACEPARENT / TRACESTATE and detaches whatever a test attached.

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"
TRACEPARENT = f"00-{TRACE_ID}-{SPAN_ID}-01"


def _current_span_context():
    return trace.get_current_span().get_span_context()


def test_inherit_trace_context_attaches_traceparent(captured_configure, monkeypatch):
    monkeypatch.setenv("TRACEPARENT", TRACEPARENT)

    telemetry.configure(service_name="haiku-ingester", inherit_trace_context=True)

    span_context = _current_span_context()
    assert format(span_context.trace_id, "032x") == TRACE_ID
    assert format(span_context.span_id, "016x") == SPAN_ID
    assert span_context.is_remote


def test_trace_context_not_inherited_by_default(captured_configure, monkeypatch):
    monkeypatch.setenv("TRACEPARENT", TRACEPARENT)

    telemetry.configure(service_name="haiku-rag-app")

    assert telemetry._env_context_token is None
    assert not _current_span_context().is_valid


@pytest.mark.parametrize(
    "value",
    [None, "", "   ", "not-a-traceparent", f"00-{'0' * 32}-{SPAN_ID}-01"],
    ids=["absent", "empty", "blank", "malformed", "invalid-trace-id"],
)
def test_attach_env_context_ignores_missing_or_bad_traceparent(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("TRACEPARENT", value)

    assert telemetry.attach_env_context() is False
    assert telemetry._env_context_token is None
    assert not _current_span_context().is_valid


def test_attach_env_context_carries_tracestate(monkeypatch):
    monkeypatch.setenv("TRACEPARENT", TRACEPARENT)
    monkeypatch.setenv("TRACESTATE", "vendor=value,other=thing")

    assert telemetry.attach_env_context() is True

    trace_state = _current_span_context().trace_state
    assert trace_state.get("vendor") == "value"
    assert trace_state.get("other") == "thing"


def test_attach_env_context_is_idempotent(monkeypatch):
    monkeypatch.setenv("TRACEPARENT", TRACEPARENT)
    assert telemetry.attach_env_context() is True
    token = telemetry._env_context_token

    assert telemetry.attach_env_context() is True
    assert telemetry._env_context_token is token


def test_attach_env_context_never_raises(monkeypatch):
    monkeypatch.setenv("TRACEPARENT", TRACEPARENT)

    def _boom(self, carrier):
        raise RuntimeError("propagator broke")

    monkeypatch.setattr(telemetry.TraceContextTextMapPropagator, "extract", _boom)

    assert telemetry.attach_env_context() is False
    assert telemetry._env_context_token is None


def test_spans_under_asyncio_run_have_the_traceparent_parent(monkeypatch):
    """run-batch creates its sweep span inside asyncio.run(), which copies the
    context current at that moment into its task. Attaching first must make
    the traceparent span the parent there."""
    monkeypatch.setenv("TRACEPARENT", TRACEPARENT)
    assert telemetry.attach_env_context() is True

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    async def _run():
        with tracer.start_as_current_span("ingester.poller.sweep"):
            pass

    asyncio.run(_run())

    (span,) = exporter.get_finished_spans()
    assert format(span.context.trace_id, "032x") == TRACE_ID
    assert span.parent is not None
    assert format(span.parent.span_id, "016x") == SPAN_ID
    assert span.parent.is_remote


def test_attach_ignores_logfire_distributed_tracing_setting(monkeypatch):
    """logfire.configure() wraps the global propagator, and with
    distributed_tracing=False drops any extracted context. The join is
    deliberate, so it must not go through that propagator."""
    from logfire.propagate import NoExtractTraceContextPropagator
    from opentelemetry import propagate

    monkeypatch.setenv("TRACEPARENT", TRACEPARENT)
    original = propagate.get_global_textmap()
    propagate.set_global_textmap(NoExtractTraceContextPropagator(original))
    try:
        assert telemetry.attach_env_context() is True
    finally:
        propagate.set_global_textmap(original)

    assert format(_current_span_context().trace_id, "032x") == TRACE_ID


def test_logfire_failure_does_not_stop_the_attach(monkeypatch):
    monkeypatch.setenv("TRACEPARENT", TRACEPARENT)

    def _broken_configure(**kwargs):
        raise RuntimeError("logfire misconfigured")

    monkeypatch.setattr(logfire, "configure", _broken_configure)

    telemetry.configure(service_name="haiku-ingester", inherit_trace_context=True)

    assert format(_current_span_context().trace_id, "032x") == TRACE_ID
