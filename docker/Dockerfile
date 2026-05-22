# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies into a venv
# In workspace: root package (haiku.rag) includes all extras via haiku.rag-slim
# Copy workspace member directories first (needed for workspace resolution)
COPY pyproject.toml uv.lock ./
COPY haiku_rag_slim/pyproject.toml haiku_rag_slim/README.md haiku_rag_slim/LICENSE haiku_rag_slim/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra ingester

# Install the project itself (with the ingester extra so haiku-ingester is on PATH)
COPY haiku_rag_slim haiku_rag_slim/
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --no-dev --extra ingester

# Final layer
FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Set default data directory
RUN mkdir -p /data
ENV DEFAULT_DATA_DIR=/data

ENV PATH="/app/.venv/bin:$PATH"

# Expose MCP server (8001) and ingester control plane (8765) ports.
# docker-compose overrides this image's default command to run either the
# MCP server (read-only) or the ingester service.
EXPOSE 8001 8765

# Default command: read-only MCP server. The companion ingester service is
# launched via docker-compose against the same image.
CMD ["haiku-rag", "--config", "/app/haiku.rag.yaml", "--read-only", "mcp", "--port", "8001"]
