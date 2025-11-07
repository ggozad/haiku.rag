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
    uv sync --frozen --no-install-project --no-dev

# Install the project itself
COPY haiku_rag_slim haiku_rag_slim/
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --no-dev

# Final layer
FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Set default data directory
RUN mkdir -p /data
ENV DEFAULT_DATA_DIR=/data

ENV PATH="/app/.venv/bin:$PATH"

# Expose port for MCP server
EXPOSE 8001

# Run all services (monitoring, MCP)
CMD ["python", "-m", "haiku.rag.cli", "serve", "--monitor", "--mcp", "--mcp-port", "8001", "--db", "/data/haiku.rag.lancedb"]
