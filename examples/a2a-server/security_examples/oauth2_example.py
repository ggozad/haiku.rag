"""Example: Adding OAuth2 authentication to haiku.rag A2A agent.

This example demonstrates OAuth2 client credentials flow with JWT token verification.
Suitable for enterprise environments with existing OAuth2 infrastructure.

Requirements:
    uv pip install python-jose[cryptography]

Setup:
    1. Set up an OAuth2 provider (Auth0, Okta, Azure AD, Keycloak, etc.)
    2. Create an API and a machine-to-machine application
    3. Get the token URL and public key from your provider
    4. Set environment variables:
       export OAUTH2_TOKEN_URL='https://your-auth.example.com/oauth/token'
       export OAUTH2_PUBLIC_KEY='-----BEGIN PUBLIC KEY-----...'

Usage:
    python oauth2_example.py /path/to/database.lancedb

    # Get access token from your OAuth2 provider:
    TOKEN=$(curl -X POST $OAUTH2_TOKEN_URL \
      -d "grant_type=client_credentials" \
      -d "client_id=your-client-id" \
      -d "client_secret=your-client-secret" \
      -d "scope=read:documents query:documents" \
      | jq -r '.access_token')

    # Make authenticated request:
    curl -H "Authorization: Bearer $TOKEN" \
         -H "Content-Type: application/json" \
         -X POST http://localhost:8000/ \
         -d '{"jsonrpc":"2.0","method":"message/send","params":{"contextId":"test","message":{"kind":"message","role":"user","messageId":"msg-1","parts":[{"kind":"text","text":"What is Python?"}]}},"id":1}'
"""

import os
from pathlib import Path

from haiku_rag_a2a.a2a import create_a2a_app
from jose import JWTError, jwt
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

# OAuth2 Configuration
OAUTH2_TOKEN_URL = os.getenv(
    "OAUTH2_TOKEN_URL", "https://your-auth.example.com/oauth/token"
)
OAUTH2_AUTH_URL = os.getenv(
    "OAUTH2_AUTH_URL", "https://your-auth.example.com/oauth/authorize"
)
OAUTH2_PUBLIC_KEY = os.getenv("OAUTH2_PUBLIC_KEY", "")
OAUTH2_ALGORITHM = os.getenv("OAUTH2_ALGORITHM", "RS256")

# Define required scopes for each skill
SKILL_SCOPES = {
    "document-qa": ["read:documents", "query:documents"],
}


def verify_token(token: str) -> dict:
    """Verify JWT token from OAuth2 provider.

    Args:
        token: JWT token from Authorization header

    Returns:
        Dictionary with user info and scopes

    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not OAUTH2_PUBLIC_KEY:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth2 public key not configured",
        )

    try:
        payload = jwt.decode(
            token,
            OAUTH2_PUBLIC_KEY,
            algorithms=[OAUTH2_ALGORITHM],
        )

        username: str | None = payload.get("sub")
        scopes: list[str] = (
            payload.get("scope", "").split()
            if isinstance(payload.get("scope"), str)
            else payload.get("scope", [])
        )

        if username is None:
            raise credentials_exception

        return {"username": username, "scopes": scopes}

    except JWTError as e:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def check_skill_permissions(skill_id: str, credentials: dict) -> None:
    """Verify that user has required scopes for a skill.

    Args:
        skill_id: The skill being accessed
        credentials: User credentials with scopes

    Raises:
        HTTPException: If user lacks required permissions
    """
    required_scopes = SKILL_SCOPES.get(skill_id, [])
    user_scopes = credentials.get("scopes", [])

    missing_scopes = [scope for scope in required_scopes if scope not in user_scopes]

    if missing_scopes:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail=f"Missing required scopes: {', '.join(missing_scopes)} for skill: {skill_id}",
        )


def create_secure_a2a_app(db_path: Path):
    """Create A2A app with OAuth2 authentication.

    Args:
        db_path: Path to LanceDB database

    Returns:
        FastA2A application with OAuth2 security
    """
    # Create app with security declared in AgentCard
    app = create_a2a_app(
        db_path,
        security_schemes={
            "oauth2": {
                "type": "oauth2",
                "flows": {
                    "clientCredentials": {
                        "tokenUrl": OAUTH2_TOKEN_URL,
                        "scopes": {
                            "read:documents": "Read document content",
                            "query:documents": "Search and query documents",
                        },
                    }
                },
                "description": "OAuth2 client credentials flow",
            }
        },
        security=[{"oauth2": ["read:documents", "query:documents"]}],
    )

    # Add authentication middleware
    @app.middleware("http")
    async def authenticate_request(request, call_next):
        """Middleware to verify OAuth2 token on all requests."""
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
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Verify token
        try:
            credentials = verify_token(token)
            # Attach credentials to request state for use in handlers
            request.state.credentials = credentials
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
        print("Usage: python oauth2_example.py <path-to-database.lancedb>")
        sys.exit(1)

    db_path = Path(sys.argv[1])
    app = create_secure_a2a_app(db_path)

    uvicorn.run(app, host="127.0.0.1", port=8000)
