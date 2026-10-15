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
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

RUN mkdir -p /data
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "--factory", "haiku_rag_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
