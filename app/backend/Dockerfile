# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install haiku.rag-slim from workspace
COPY pyproject.toml uv.lock ./
COPY haiku_rag_slim/pyproject.toml haiku_rag_slim/README.md haiku_rag_slim/LICENSE haiku_rag_slim/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --package haiku.rag-slim

COPY haiku_rag_slim haiku_rag_slim/
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --no-dev --package haiku.rag-slim

# Install app backend dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install starlette uvicorn[standard] anthropic watchfiles

# Final layer
FROM python:3.13-slim
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY app/backend/*.py ./

RUN mkdir -p /data
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
