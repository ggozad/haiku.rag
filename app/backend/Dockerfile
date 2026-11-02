# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
COPY haiku_rag_slim/pyproject.toml haiku_rag_slim/README.md haiku_rag_slim/LICENSE haiku_rag_slim/
COPY app/backend/pyproject.toml app/backend/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --no-dev --package haiku-rag-app

COPY haiku_rag_slim haiku_rag_slim/
COPY app/backend/haiku_rag_app app/backend/haiku_rag_app/
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --no-dev --package haiku-rag-app

FROM python:3.13-slim
ARG VERSION
LABEL org.opencontainers.image.source="https://github.com/ggozad/haiku.rag" \
      org.opencontainers.image.version="${VERSION}"
WORKDIR /app

RUN addgroup --system --gid 1001 haiku && \
    adduser --system --uid 1001 --ingroup haiku haiku

COPY --from=builder --chown=haiku:haiku /app/.venv /app/.venv

# Compose seeds a new named volume from the image directory, owner included.
RUN mkdir -p /data /sessions && chown haiku:haiku /data /sessions
ENV PATH="/app/.venv/bin:$PATH"

USER haiku

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

# One worker: the session store holds runs in process.
CMD ["uvicorn", "--factory", "haiku_rag_app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
