FROM ghcr.io/ggozad/haiku.rag-slim:latest

WORKDIR /app

# Copy backend application files
COPY main.py agent.py ./

# Install additional dependencies for the example
RUN pip install --no-cache-dir \
    starlette>=0.50.0 \
    uvicorn[standard]>=0.40.0 \
    python-dotenv>=1.2.1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
