"""Supabase JWT authentication for FastAPI."""
import logging
import os
import httpx
import jwt
from jwt import PyJWK
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

security = HTTPBearer()

_supabase_url = os.getenv("SUPABASE_URL", "")
_jwks_url = f"{_supabase_url}/auth/v1/.well-known/jwks.json"
_jwks_cache: dict | None = None


def _get_signing_key(token: str):
    """Fetch the JWKS from Supabase and return the matching key for the token."""
    global _jwks_cache
    if _jwks_cache is None:
        resp = httpx.get(_jwks_url, timeout=10)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        logger.info("Fetched JWKS from Supabase")

    header = jwt.get_unverified_header(token)
    kid = header.get("kid")

    for key_data in _jwks_cache.get("keys", []):
        if key_data.get("kid") == kid:
            return PyJWK(key_data).key

    # kid not found – refresh cache once and retry
    resp = httpx.get(_jwks_url, timeout=10)
    resp.raise_for_status()
    _jwks_cache = resp.json()

    for key_data in _jwks_cache.get("keys", []):
        if key_data.get("kid") == kid:
            return PyJWK(key_data).key

    raise ValueError(f"No matching key found for kid={kid}")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate Supabase JWT and return user_id."""
    token = credentials.credentials
    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "HS256")

        signing_key = _get_signing_key(token)

        payload = jwt.decode(
            token,
            signing_key,
            algorithms=[alg],
            audience="authenticated",
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no sub claim")
        return user_id
    except HTTPException:
        raise
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail=f"Authentication failed: {e}")
