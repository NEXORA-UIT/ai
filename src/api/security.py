"""Authentication dependencies for private AI service routes."""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.config import (
    SERVICE_API_KEY_ENV,
    SERVICE_API_KEY_MISSING_DETAIL,
    SERVICE_API_KEY_SCHEME_NAME,
    SERVICE_API_KEY_UNCONFIGURED_DETAIL,
)

_service_bearer = HTTPBearer(auto_error=False, scheme_name=SERVICE_API_KEY_SCHEME_NAME)
_service_bearer_dependency = Security(_service_bearer)


def require_service_key(
    credentials: HTTPAuthorizationCredentials | None = _service_bearer_dependency,
) -> None:
    """Require the configured service key and fail closed when it is absent."""
    expected = os.environ.get(SERVICE_API_KEY_ENV)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=SERVICE_API_KEY_UNCONFIGURED_DETAIL,
        )

    provided = credentials.credentials if credentials is not None else ""
    if not secrets.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=SERVICE_API_KEY_MISSING_DETAIL,
            headers={"WWW-Authenticate": "Bearer"},
        )
