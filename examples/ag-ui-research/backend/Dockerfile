FROM ghcr.io/ggozad/haiku.rag:latest

WORKDIR /app

# Copy backend application files
COPY agent.py main.py ./
COPY pyproject.toml ./

# Install backend dependencies
# Note: haiku-rag is already installed in the base image
RUN pip install --no-cache-dir \
    starlette>=0.45.2 \
    uvicorn[standard]>=0.34.2 \
    pydantic-ai-slim[ag-ui,openai]>=1.1.0 \
    python-dotenv>=1.0.1

EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
