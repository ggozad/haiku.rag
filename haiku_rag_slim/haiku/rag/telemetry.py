import logging
import os
from importlib import metadata
from typing import Literal

from logfire import Logfire, attach_context, get_context
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

_logger = logging.getLogger(__name__)

# Scoped Logfire instance — every span emitted through `logfire.span(...)`
# on this object carries `instrumentation_scope.name = "haiku.rag"` instead
# of the default "logfire". The scope is OTel's identifier for *which
# library* produced the span, separate from `service.name` which is the
# running process. Downstream consumers (Logfire UI saved views, OTel
# collectors, alert rules) can then filter on `scope.name = haiku.rag`
# rather than catching every span the SDK ever exports.
#
# Cross-library instrumentations (pydantic-ai, FastAPI, OpenAI) keep their
# own scopes — this only retags the spans WE write.
logfire = Logfire(otel_scope="haiku.rag")

# Token for the context attach_env_context() attached. Held for the life of
# the process and never detached: the parent trace is meant to cover every
# span this process creates.
_env_context_token: object | None = None


def attach_env_context() -> bool:
    """Join the trace a parent process passed in `TRACEPARENT` (and
    optionally `TRACESTATE`), W3C trace-context headers carried in the
    environment. Neither the OpenTelemetry SDK nor Logfire reads them, so
    without this a subprocess always starts a trace of its own.

    Must run in the main thread before `asyncio.run`, which copies the
    current context into its task. Returns True when a parent context is
    attached (now or by an earlier call), False when there is none to
    attach. Never raises.
    """
    global _env_context_token
    try:
        if _env_context_token is not None:
            return True
        traceparent = os.environ.get("TRACEPARENT", "").strip()
        if not traceparent:
            return False
        carrier = {"traceparent": traceparent}
        tracestate = os.environ.get("TRACESTATE", "").strip()
        if tracestate:
            carrier["tracestate"] = tracestate
        # The W3C propagator itself, not the global one: logfire.configure()
        # wraps that to warn on (or, with distributed_tracing=False, drop)
        # an extracted context, and this join is deliberate.
        ctx = TraceContextTextMapPropagator().extract(carrier)
        if not trace.get_current_span(ctx).get_span_context().is_valid:
            _logger.debug("Ignoring malformed TRACEPARENT %r", traceparent)
            return False
        # Not logfire's attach_context: that is a context manager which
        # detaches on exit, and this context lasts the whole process.
        _env_context_token = otel_context.attach(ctx)
        return True
    except Exception:
        _logger.debug("Could not attach TRACEPARENT context", exc_info=True)
        return False


def configure(
    *,
    service_name: str | None = None,
    console: Literal[False] | None = False,
    scrubbing: Literal[False] | None = None,
    inherit_trace_context: bool = False,
) -> None:
    """Configure Logfire and enable pydantic-ai instrumentation for the
    running process. Each CLI entry point calls this once at startup.
    Silently no-ops on failure so a missing/misconfigured LOGFIRE_TOKEN
    never crashes the app.

    - service_name: the default name for this process in the Logfire UI
      (e.g. "haiku-ingester"). The OTEL_SERVICE_NAME / LOGFIRE_SERVICE_NAME
      env vars, when set, take precedence so operators can distinguish
      concurrent processes.
    - console: False (default) suppresses span lines on stderr so they
      don't interleave with RichHandler logs. Pass None to let logfire
      decide (its own default applies).
    - scrubbing: None (default) keeps logfire's secret scrubbing on. Pass
      False to disable it when span content legitimately contains tokens
      that trip the scrubber (e.g. eval answer text).
    - inherit_trace_context: True joins the parent trace passed in
      `TRACEPARENT` / `TRACESTATE` (see attach_env_context). Off by default:
      only one-shot commands opt in, since a long-running process would put
      its whole lifetime into one trace that never ends.
    """
    try:
        import logfire as _lf

        # An explicit service_name arg would beat the env vars in logfire's
        # precedence; deferring to None when an env var is set lets the
        # operator's OTEL_SERVICE_NAME / LOGFIRE_SERVICE_NAME win over our
        # per-process default.
        env_service = os.environ.get("OTEL_SERVICE_NAME") or os.environ.get(
            "LOGFIRE_SERVICE_NAME"
        )
        try:
            service_version = metadata.version("haiku.rag-slim")
        except metadata.PackageNotFoundError:  # pragma: no cover
            service_version = None

        _lf.configure(
            service_name=None if env_service else service_name,
            service_version=service_version,
            send_to_logfire="if-token-present",
            console=console,
            scrubbing=scrubbing,
        )
        _lf.instrument_pydantic_ai()
    except Exception:
        pass

    # Outside the try above: attaching needs no tracer provider, so a
    # Logfire failure must not skip it.
    if inherit_trace_context:
        attach_env_context()


__all__ = [
    "attach_context",
    "attach_env_context",
    "configure",
    "get_context",
    "logfire",
]
