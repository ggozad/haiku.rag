"""Example: Using GitHub Personal Access Tokens for authentication.

This is a simplified OAuth2 example that uses GitHub Personal Access Tokens.
It's much easier to set up than full OAuth2 and perfect for testing.

Setup:
    1. Go to https://github.com/settings/tokens
    2. Click "Generate new token (classic)"
    3. Give it a name and select scopes
    4. Copy the generated token

Usage:
    export GITHUB_TOKENS="your_github_token_here"
    python oauth2_github.py /path/to/database.lancedb

    # Make authenticated request:
    curl -H "Authorization: Bearer ghp_your_token" \
         -H "Content-Type: application/json" \
         -X POST http://localhost:8000/ \
         -d '{"jsonrpc":"2.0","method":"message/send","params":{"contextId":"test","message":{"kind":"message","role":"user","messageId":"msg-1","parts":[{"kind":"text","text":"What is Python?"}]}},"id":1}'
"""

import os
from pathlib import Path

import httpx
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED

from haiku.rag.a2a import create_a2a_app

# Configuration
GITHUB_API_URL = "https://api.github.com"
ALLOWED_TOKENS = (
    set(os.getenv("GITHUB_TOKENS", "").split(","))
    if os.getenv("GITHUB_TOKENS")
    else set()
)


async def verify_github_token(token: str) -> dict:
    """Verify GitHub Personal Access Token by calling GitHub API.

    Args:
        token: GitHub Personal Access Token (starts with ghp_)

    Returns:
        Dictionary with user info

    Raises:
        HTTPException: If token is invalid
    """
    if not token.startswith("ghp_") and not token.startswith("github_pat_"):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid GitHub token format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # If we have a list of allowed tokens, check against it
    if ALLOWED_TOKENS and token not in ALLOWED_TOKENS:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Token not in allowed list",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify token with GitHub API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{GITHUB_API_URL}/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=5.0,
            )

            if response.status_code == 401:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired GitHub token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail=f"GitHub API error: {response.status_code}",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            user_data = response.json()
            return {
                "username": user_data.get("login"),
                "email": user_data.get("email"),
                "name": user_data.get("name"),
                "github_id": user_data.get("id"),
            }

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="GitHub API timeout",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail=f"Failed to verify token with GitHub: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )


def create_secure_a2a_app(db_path: Path):
    """Create A2A app with GitHub token authentication.

    Args:
        db_path: Path to LanceDB database

    Returns:
        FastA2A application with GitHub authentication
    """
    # Create app with security declared in AgentCard
    app = create_a2a_app(
        db_path,
        security_schemes={
            "githubAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": "GitHub Personal Access Token authentication",
            }
        },
        security=[{"githubAuth": []}],
    )

    @app.middleware("http")
    async def authenticate_request(request, call_next):
        """Middleware to verify GitHub token on all requests."""
        # Skip authentication for well-known endpoints
        if request.url.path in [
            "/.well-known/agent-card.json",
            "/health",
            "/docs",
            "/openapi.json",
        ]:
            return await call_next(request)

        # Get token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid Authorization header"},
                headers={"WWW-Authenticate": 'Bearer realm="GitHub"'},
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Verify token
        try:
            user_data = await verify_github_token(token)
            # Attach user data to request state
            request.state.user = user_data
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
        print("Usage: python oauth2_github.py <path-to-database.lancedb>")
        sys.exit(1)

    db_path = Path(sys.argv[1])
    app = create_secure_a2a_app(db_path)

    uvicorn.run(app, host="127.0.0.1", port=8000)
