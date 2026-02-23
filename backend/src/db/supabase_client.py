"""Singleton Supabase client for backend."""
import os
import logging
from supabase import create_client, Client

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_supabase_client() -> Client:
    """Get or create the Supabase client singleton.
    
    Uses SUPABASE_SERVICE_ROLE_KEY (bypasses RLS) since
    the backend handles auth and scopes queries by user_id.
    """
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment"
            )
        _client = create_client(url, key)
        logger.info(f"Supabase client initialized for {url}")
    return _client
