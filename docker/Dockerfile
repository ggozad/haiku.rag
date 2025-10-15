# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY . /app

# Install haiku.rag with all extras in development mode
RUN uv pip install --system -e ".[voyageai,mxbai,a2a]"

# Create data directory for the database
RUN mkdir -p /data

# Default database path
ENV HAIKU_RAG_DB_PATH=/data/haiku.rag.lancedb

# Expose ports for MCP and A2A
EXPOSE 8000 8001

# Default command runs all services (monitoring, MCP, A2A)
CMD ["haiku-rag", "serve", "--monitor", "--mcp", "--mcp-port", "8001", "--a2a", "--a2a-host", "0.0.0.0", "--a2a-port", "8000", "--db", "/data/haiku.rag.lancedb"]
