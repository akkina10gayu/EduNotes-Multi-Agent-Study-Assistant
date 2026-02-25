"""Document store backed by Supabase + pgvector.

Replaces the legacy VectorStore (ChromaDB) and DocumentProcessor with a
unified interface for document ingestion, chunking, embedding, and
semantic search using Supabase's pgvector integration.
"""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singleton for the embedding model
# ---------------------------------------------------------------------------
_embedding_model = None


def _get_embedding_model():
    """Return the SentenceTransformer model, loading it on first call."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded embedding model all-MiniLM-L6-v2 (384-dim)")
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise RuntimeError(
                "sentence-transformers is required but could not be loaded"
            ) from exc
    return _embedding_model


def _generate_embedding(text: str) -> list[float]:
    """Generate a 384-dimensional embedding for *text*."""
    model = _get_embedding_model()
    embedding = model.encode(text)
    return embedding.tolist()


def _get_text_splitter():
    """Return a configured RecursiveCharacterTextSplitter."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_document(
    user_id: str,
    title: str,
    topic: str,
    content: str,
    source: str = "",
    url: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Ingest a document: store it, chunk the content, embed, and index.

    Returns the newly created document id.
    """
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    # 1. Insert the parent document row
    doc_payload = {
        "user_id": user_id,
        "title": title,
        "topic": topic,
        "content": content,
        "source": source,
        "url": url,
        "metadata": metadata or {},
    }

    logger.info("Inserting document '%s' for user %s", title, user_id)
    result = supabase.table("documents").insert(doc_payload).execute()
    doc_id = result.data[0]["id"]
    logger.info("Document inserted with id %s", doc_id)

    # 2. Chunk the content
    splitter = _get_text_splitter()
    chunks = splitter.split_text(content)
    logger.info("Split document %s into %d chunks", doc_id, len(chunks))

    # 3. Generate embeddings and insert chunks
    chunk_rows = []
    for idx, chunk_text in enumerate(chunks):
        embedding = _generate_embedding(chunk_text)
        chunk_rows.append(
            {
                "document_id": doc_id,
                "user_id": user_id,
                "content": chunk_text,
                "embedding": embedding,
                "chunk_index": idx,
                "metadata": {
                    "document_id": doc_id,
                    "title": title,
                    "topic": topic,
                    "source": source,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                },
            }
        )

    if chunk_rows:
        supabase.table("document_chunks").insert(chunk_rows).execute()
        logger.info("Inserted %d chunks for document %s", len(chunk_rows), doc_id)

    return doc_id


def search(
    user_id: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """Semantic search across a user's document chunks via pgvector RPC.

    Returns a list of dicts with keys: content, metadata, similarity.
    """
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    query_embedding = _generate_embedding(query)
    logger.info(
        "Searching for user %s — query: '%s' (top_k=%d, threshold=%.2f)",
        user_id,
        query[:80],
        top_k,
        threshold,
    )

    result = supabase.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": top_k,
            "p_user_id": user_id,
        },
    ).execute()

    matches: list[dict[str, Any]] = []
    for row in result.data or []:
        matches.append(
            {
                "content": row.get("content", ""),
                "metadata": row.get("metadata", {}),
                "similarity": row.get("similarity", 0.0),
            }
        )

    logger.info("Search returned %d results", len(matches))
    return matches


def list_documents(user_id: str) -> list[dict[str, Any]]:
    """List all documents belonging to *user_id*."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    result = (
        supabase.table("documents")
        .select("id, title, topic, source, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    logger.info("Listed %d documents for user %s", len(result.data), user_id)
    return result.data


def get_document(user_id: str, doc_id: str) -> dict[str, Any] | None:
    """Retrieve a single document with full content."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    result = (
        supabase.table("documents")
        .select("id, title, topic, source, url, content, metadata, created_at")
        .eq("id", doc_id)
        .eq("user_id", user_id)
        .execute()
    )

    if result.data:
        logger.info("Retrieved document %s for user %s", doc_id, user_id)
        return result.data[0]

    logger.warning("Document %s not found for user %s", doc_id, user_id)
    return None


def delete_document(user_id: str, doc_id: str) -> None:
    """Delete a document and its chunks (cascade)."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    logger.info("Deleting document %s for user %s", doc_id, user_id)
    supabase.table("documents").delete().eq("id", doc_id).eq(
        "user_id", user_id
    ).execute()
    logger.info("Document %s deleted", doc_id)


def get_collection_stats(user_id: str) -> dict[str, int]:
    """Return document and chunk counts for *user_id*."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    docs_result = (
        supabase.table("documents")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .execute()
    )
    chunks_result = (
        supabase.table("document_chunks")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .execute()
    )

    stats = {
        "total_documents": docs_result.count or 0,
        "total_chunks": chunks_result.count or 0,
    }
    logger.info("Collection stats for user %s: %s", user_id, stats)
    return stats


def get_unique_topics(user_id: str) -> list[str]:
    """Return distinct topic strings for *user_id*."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    result = (
        supabase.table("documents")
        .select("topic")
        .eq("user_id", user_id)
        .execute()
    )

    topics = sorted({row["topic"] for row in result.data if row.get("topic")})
    logger.info("Found %d unique topics for user %s", len(topics), user_id)
    return topics


def search_documents_semantic(
    user_id: str,
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Semantic search returning unique parent documents (not individual chunks).

    Groups chunk results by document_id and keeps the best similarity score
    per document.  Returns up to *limit* document metadata dicts.
    """
    # Fetch more chunks than the final limit so we have enough unique docs
    chunks = search(user_id, query, top_k=limit * 3, threshold=0.5)

    # Group by document_id, keeping the best score
    best_by_doc: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        doc_id = meta.get("document_id") or meta.get("id")
        if not doc_id:
            continue

        similarity = chunk.get("similarity", 0.0)
        if doc_id not in best_by_doc or similarity > best_by_doc[doc_id]["similarity"]:
            best_by_doc[doc_id] = {
                "document_id": doc_id,
                "title": meta.get("title", ""),
                "topic": meta.get("topic", ""),
                "source": meta.get("source", ""),
                "similarity": similarity,
            }

    # Sort by similarity descending and limit
    results = sorted(best_by_doc.values(), key=lambda d: d["similarity"], reverse=True)[
        :limit
    ]
    logger.info(
        "Semantic doc search for user %s returned %d unique documents",
        user_id,
        len(results),
    )
    return results
