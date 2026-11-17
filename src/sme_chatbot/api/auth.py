"""``get_identity`` FastAPI dependency — ``AUTH_MODE=dev|jwt`` (plan §4.2).

- ``dev`` (default): no ``Authorization`` required; ``user_id`` comes from the
  optional ``X-User-Id`` header; identity is unrestricted.
- ``jwt``: validates an ``Authorization: Bearer`` token against Entra ID via
  ``python-jose``, fetching the tenant's JWKS lazily (and caching it) rather
  than at import time, so importing this module never makes a network call
  and dev-mode usage is never affected. Claim→scope mapping is a stub for
  this phase: any validated token yields an unrestricted ``RequestIdentity``.

Handlers/tools never see headers or tokens directly — only the
``RequestIdentity`` returned here.
"""

from __future__ import annotations

import logging
import time

import httpx
from fastapi import Header, HTTPException, status
from jose import jwt
from jose.exceptions import JOSEError

from sme_chatbot import config
from sme_chatbot.core.events import RequestIdentity

logger = logging.getLogger(__name__)

_JWKS_TTL_SECONDS = 3600

# Lazy, process-wide JWKS cache. Populated on first ``jwt``-mode request.
_jwks_cache: dict | None = None
_jwks_cache_time: float = 0.0


def _jwks_url() -> str:
    return f"https://login.microsoftonline.com/{config.JWT_TENANT_ID}/discovery/v2.0/keys"


def _fetch_jwks() -> dict:
    resp = httpx.get(_jwks_url(), timeout=10.0)
    resp.raise_for_status()
    return resp.json()


def _get_jwks(force_refresh: bool = False) -> dict:
    """Return the cached JWKS, fetching (or re-fetching) it lazily."""
    global _jwks_cache, _jwks_cache_time
    now = time.monotonic()
    stale = _jwks_cache is None or (now - _jwks_cache_time) > _JWKS_TTL_SECONDS
    if force_refresh or stale:
        _jwks_cache = _fetch_jwks()
        _jwks_cache_time = now
    return _jwks_cache


def _find_key(jwks: dict, kid: str | None) -> dict | None:
    return next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)


def _validate_jwt(token: str) -> dict:
    try:
        header = jwt.get_unverified_header(token)
    except JOSEError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token header: {exc}",
        ) from exc

    kid = header.get("kid")
    try:
        jwks = _get_jwks()
        key = _find_key(jwks, kid)
        if key is None:
            # Key rotation: force one refresh before giving up.
            jwks = _get_jwks(force_refresh=True)
            key = _find_key(jwks, kid)
    except httpx.HTTPError as exc:
        logger.exception("Failed to fetch Entra JWKS")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not fetch signing keys: {exc}",
        ) from exc

    if key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown signing key (kid)",
        )

    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[key.get("alg", "RS256")],
            audience=config.JWT_AUDIENCE,
        )
    except JOSEError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        ) from exc
    return claims


def get_identity(
    authorization: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> RequestIdentity:
    """FastAPI dependency resolving the caller's :class:`RequestIdentity`."""
    if config.AUTH_MODE == "jwt":
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization: Bearer token",
            )
        token = authorization.split(" ", 1)[1].strip()
        claims = _validate_jwt(token)
        user_id = claims.get("oid") or claims.get("sub")
        # Claim -> scope mapping stub (report §12.3 fills this in later):
        # a validated token always yields an unrestricted identity for now.
        return RequestIdentity(user_id=user_id, permitted_document_ids=None, scopes={})

    # AUTH_MODE=dev (default): no auth required.
    return RequestIdentity(user_id=x_user_id, permitted_document_ids=None, scopes={})
