"""Supabase Storage wrapper for PDF uploads and Anki exports.

Provides helpers for uploading, downloading, and deleting files in
Supabase Storage buckets (``pdfs`` and ``exports``).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Bucket names
BUCKET_PDFS = "pdfs"
BUCKET_EXPORTS = "exports"

# Signed-URL expiry in seconds (1 hour)
_SIGNED_URL_EXPIRY = 3600


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upload_pdf(user_id: str, filename: str, content_bytes: bytes) -> str:
    """Upload a PDF file to ``pdfs/{user_id}/{filename}``.

    Returns the storage path of the uploaded file.
    """
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()
    path = f"{user_id}/{filename}"

    logger.info("Uploading PDF to %s/%s", BUCKET_PDFS, path)
    try:
        supabase.storage.from_(BUCKET_PDFS).upload(
            path,
            content_bytes,
            file_options={"content-type": "application/pdf"},
        )
        logger.info("PDF uploaded successfully: %s/%s", BUCKET_PDFS, path)
    except Exception as exc:
        logger.error("Failed to upload PDF %s/%s: %s", BUCKET_PDFS, path, exc)
        raise

    return f"{BUCKET_PDFS}/{path}"


def upload_export(user_id: str, filename: str, content: str) -> str:
    """Upload a text export to ``exports/{user_id}/{filename}``.

    Returns a signed URL valid for 1 hour.
    """
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()
    path = f"{user_id}/{filename}"

    logger.info("Uploading export to %s/%s", BUCKET_EXPORTS, path)
    try:
        supabase.storage.from_(BUCKET_EXPORTS).upload(
            path,
            content.encode("utf-8"),
            file_options={"content-type": "text/plain"},
        )
        logger.info("Export uploaded successfully: %s/%s", BUCKET_EXPORTS, path)
    except Exception as exc:
        logger.error("Failed to upload export %s/%s: %s", BUCKET_EXPORTS, path, exc)
        raise

    try:
        signed = supabase.storage.from_(BUCKET_EXPORTS).create_signed_url(
            path, _SIGNED_URL_EXPIRY
        )
        url: str = signed.get("signedURL") or signed.get("signedUrl", "")
        logger.info("Generated signed URL for export %s/%s", BUCKET_EXPORTS, path)
        return url
    except Exception as exc:
        logger.error(
            "Failed to create signed URL for export %s/%s: %s",
            BUCKET_EXPORTS,
            path,
            exc,
        )
        raise


def get_pdf_url(user_id: str, filename: str) -> str:
    """Return a signed URL for the PDF at ``pdfs/{user_id}/{filename}``.

    The URL is valid for 1 hour.
    """
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()
    path = f"{user_id}/{filename}"

    logger.info("Creating signed URL for PDF %s/%s", BUCKET_PDFS, path)
    try:
        signed = supabase.storage.from_(BUCKET_PDFS).create_signed_url(
            path, _SIGNED_URL_EXPIRY
        )
        url: str = signed.get("signedURL") or signed.get("signedUrl", "")
        logger.info("Signed URL created for PDF %s/%s", BUCKET_PDFS, path)
        return url
    except Exception as exc:
        logger.error(
            "Failed to create signed URL for PDF %s/%s: %s",
            BUCKET_PDFS,
            path,
            exc,
        )
        raise


def delete_pdf(user_id: str, filename: str) -> None:
    """Delete the PDF at ``pdfs/{user_id}/{filename}``."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()
    path = f"{user_id}/{filename}"

    logger.info("Deleting PDF %s/%s", BUCKET_PDFS, path)
    try:
        supabase.storage.from_(BUCKET_PDFS).remove([path])
        logger.info("PDF deleted: %s/%s", BUCKET_PDFS, path)
    except Exception as exc:
        logger.error("Failed to delete PDF %s/%s: %s", BUCKET_PDFS, path, exc)
        raise
