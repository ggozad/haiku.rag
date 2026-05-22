from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Bearer-token gate. When the app has no auth_token configured, all
    requests are allowed (and a warning was logged at startup). Otherwise
    the Authorization header's bearer must match exactly."""
    expected = getattr(request.app.state, "auth_token", None)
    if expected is None:
        return
    if creds is None or creds.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
