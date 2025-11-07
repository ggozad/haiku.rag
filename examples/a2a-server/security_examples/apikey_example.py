"""Example: Adding API Key authentication to haiku.rag A2A agent.

Simple header-based authentication suitable for internal services and development.
Perfect for getting started with A2A authentication.

Setup:
    # Run with default key
    python apikey_example.py /path/to/database.lancedb

    # Or use your own key
    export API_KEY='your-secret-key'
    python apikey_example.py /path/to/database.lancedb

Usage:
    # Make authenticated request (default key is demo-key-12345)
    curl -H "X-API-Key: demo-key-12345" \
         -H "Content-Type: application/json" \
         -X POST http://localhost:8000/ \
         -d '{"jsonrpc":"2.0","method":"message/send","params":{"contextId":"test","message":{"kind":"message","role":"user","messageId":"msg-1","parts":[{"kind":"text","text":"What is Python?"}]}},"id":1}'
"""

import os
from pathlib import Path

from haiku_rag_a2a.a2a import create_a2a_app
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED

# API Key Configuration - In production, use environment variables or a secure key store
API_KEY_NAME = "X-API-Key"
VALID_API_KEY = os.getenv("API_KEY", "demo-key-12345")


def verify_api_key(api_key: str | None) -> str:
    """Verify API key from request header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        The verified API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": f'ApiKey realm="{API_KEY_NAME}"'},
        )

    if api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": f'ApiKey realm="{API_KEY_NAME}"'},
        )

    return api_key


def create_secure_a2a_app(db_path: Path):
    """Create A2A app with API key authentication.

    Args:
        db_path: Path to LanceDB database

    Returns:
        FastA2A application with API key security
    """
    # Create app with security declared in AgentCard
    app = create_a2a_app(
        db_path,
        security_schemes={
            "apiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": API_KEY_NAME,
                "description": "API key authentication",
            }
        },
        security=[{"apiKeyAuth": []}],
    )

    # Add authentication middleware
    @app.middleware("http")
    async def authenticate_request(request, call_next):
        """Middleware to verify API key on all requests."""
        # Skip authentication for well-known endpoints
        if request.url.path in [
            "/.well-known/agent-card.json",
            "/health",
            "/docs",
            "/openapi.json",
        ]:
            return await call_next(request)

        # Verify API key
        api_key = request.headers.get(API_KEY_NAME)
        try:
            verify_api_key(api_key)
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail},
                headers=e.headers or {},
            )

        # Continue with request
        return await call_next(request)

    return app


if __name__ == "__main__":
    import sys

    import uvicorn

    if len(sys.argv) < 2:
        print("Usage: python apikey_example.py <path-to-database.lancedb>")
        sys.exit(1)

    db_path = Path(sys.argv[1])
    app = create_secure_a2a_app(db_path)

    uvicorn.run(app, host="127.0.0.1", port=8000)
