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
) -> None:
    """Configure Logfire and enable pydantic-ai instrumentation for the
    running process. Each CLI entry point calls this once at startup.
    Silently no-ops on failure so a missing/misconfigured LOGFIRE_TOKEN
    never crashes the app.

    - service_name: identifies the process in the Logfire UI (e.g.
      "haiku-ingester"). Falls back to logfire's default when None.
    - console: False (default) suppresses span lines on stderr so they
      don't interleave with RichHandler logs. Pass None to let logfire
      decide (its own default applies).
    """
    try:
        import logfire as _lf

        _lf.configure(
            service_name=service_name,
            send_to_logfire="if-token-present",
            console=console,
        )
        _lf.instrument_pydantic_ai()
    except Exception:  # pragma: no cover
        pass


__all__ = ["attach_context", "configure", "get_context", "logfire"]
