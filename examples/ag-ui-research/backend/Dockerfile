FROM ghcr.io/ggozad/haiku.rag:latest

WORKDIR /app

# Copy backend application files
COPY agent.py main.py ./
COPY pyproject.toml uv.lock ./

# Install backend dependencies into the existing haiku.rag venv
RUN uv sync --frozen --no-dev

EXPOSE 8000

# Run with uvicorn
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
