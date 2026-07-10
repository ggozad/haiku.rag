import os
from importlib import metadata
from typing import Literal

from logfire import Logfire, attach_context, get_context

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


def configure(
    *,
    service_name: str | None = None,
    console: Literal[False] | None = False,
    scrubbing: Literal[False] | None = None,
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
    except Exception:  # pragma: no cover
        pass


__all__ = ["attach_context", "configure", "get_context", "logfire"]
