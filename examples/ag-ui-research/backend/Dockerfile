FROM ghcr.io/ggozad/haiku.rag:latest

WORKDIR /app

# Copy backend application files
COPY main.py agent.py ./

# Install additional dependencies for the example
# Note: haiku-rag-slim is already installed in the base image
RUN pip install --no-cache-dir \
    starlette>=0.45.2 \
    uvicorn[standard]>=0.34.2 \
    python-dotenv>=1.0.1

EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
