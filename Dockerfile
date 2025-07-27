FROM python:3.11-slim

# Install uv using pip instead of the shell installer - to avoid platform issues
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install the package
RUN uv pip install --system .

# Create directories for mounts
RUN mkdir -p /inbound /database /ollama/models

# Set environment variables
ENV MONITOR_DIRECTORIES="/inbound"
ENV EMBEDDINGS_PROVIDER="ollama"
ENV EMBEDDINGS_MODEL="mxbai-embed-large"
ENV EMBEDDINGS_VECTOR_DIM=1024

# Expose port
EXPOSE 8000

# Default command
CMD ["uv", "run", "haiku-rag", "serve", "--db", "/database/haiku-rag.db"]