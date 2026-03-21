"""
Integration tests for all FastAPI endpoints.

Tests marked with @pytest.mark.skipif run only when the relevant API key is
present in the environment. All other tests run unconditionally and cover
input validation, health checks, and endpoints that require no LLM.

Run from the project root:
    pytest tests/integration/test_api.py -v
"""
import os
import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health and root
# ---------------------------------------------------------------------------

def test_health_returns_200():
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_response_structure():
    data = client.get("/api/v1/health").json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "EduNotes" in response.json()["message"]


# ---------------------------------------------------------------------------
# Stats and dashboard
# ---------------------------------------------------------------------------

def test_stats_returns_200():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200


def test_stats_response_has_required_keys():
    data = client.get("/api/v1/stats").json()
    assert "knowledge_base" in data
    assert "agents" in data


def test_dashboard_stats_returns_200():
    response = client.get("/api/v1/dashboard-stats")
    assert response.status_code == 200


def test_dashboard_stats_structure():
    data = client.get("/api/v1/dashboard-stats").json()
    assert "healthy" in data
    assert "kb_documents" in data
    assert "flashcard_sets" in data
    assert "current_streak" in data


# ---------------------------------------------------------------------------
# Knowledge base search (no LLM needed)
# ---------------------------------------------------------------------------

def test_search_kb_returns_200():
    response = client.post("/api/v1/search-kb", json={"query": "machine learning"})
    assert response.status_code == 200


def test_search_kb_response_structure():
    data = client.post("/api/v1/search-kb", json={"query": "machine learning"}).json()
    assert data["success"] is True
    assert "results" in data
    assert "count" in data
    assert isinstance(data["results"], list)


def test_search_kb_count_matches_results():
    data = client.post("/api/v1/search-kb", json={"query": "neural networks"}).json()
    assert data["count"] == len(data["results"])


# ---------------------------------------------------------------------------
# Generate notes — input validation (no LLM, these always run)
# ---------------------------------------------------------------------------

def test_generate_notes_empty_query_returns_400():
    response = client.post("/api/v1/generate-notes", json={"query": ""})
    assert response.status_code == 400


def test_generate_notes_too_short_returns_400():
    response = client.post("/api/v1/generate-notes", json={"query": "ab"})
    assert response.status_code == 400


def test_generate_notes_too_long_returns_400():
    response = client.post("/api/v1/generate-notes", json={"query": "a" * 100001})
    assert response.status_code == 400


def test_generate_notes_invalid_url_returns_400():
    response = client.post(
        "/api/v1/generate-notes",
        json={"query": "http://not a valid url!!"}
    )
    assert response.status_code == 400


def test_generate_notes_missing_query_returns_422():
    # No query field at all — Pydantic validation error
    response = client.post("/api/v1/generate-notes", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Generate notes — LLM-dependent (skip if no API key)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not configured"
)
def test_generate_notes_topic_success():
    response = client.post("/api/v1/generate-notes", json={
        "query": "machine learning",
        "summarization_mode": "paragraph_summary",
        "summary_length": "brief",
        "search_mode": "web_search",
        "save_to_kb": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["notes"]) > 0
    assert data["query_type"] in ("topic", "text")


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not configured"
)
def test_generate_notes_important_points_mode():
    response = client.post("/api/v1/generate-notes", json={
        "query": "neural networks",
        "summarization_mode": "important_points",
        "summary_length": "brief",
        "search_mode": "web_search",
        "save_to_kb": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["notes"]) > 0


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not configured"
)
def test_generate_notes_from_direct_text():
    text = (
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn automatically from data. It uses statistical techniques "
        "to improve performance over time without being explicitly programmed. "
        "Common algorithms include linear regression, decision trees, and neural networks. "
        "Applications span healthcare, finance, robotics, and natural language processing. "
        "The field is growing rapidly due to increases in data availability and computing power."
    ) * 2  # ensure over 500 chars to trigger TEXT mode
    response = client.post("/api/v1/generate-notes", json={
        "query": text,
        "summarization_mode": "paragraph_summary",
        "summary_length": "brief",
        "save_to_kb": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["query_type"] == "text"


# ---------------------------------------------------------------------------
# Chat endpoint — LLM-dependent (skip if no chat provider key)
# ---------------------------------------------------------------------------

_has_chat_provider = bool(
    os.getenv("GEMINI_API_KEY") or os.getenv("CEREBRAS_API_KEY")
)


@pytest.mark.skipif(not _has_chat_provider, reason="No chat provider configured")
def test_chat_message_basic():
    response = client.post("/api/v1/chat/message", json={
        "message": "What is supervised learning?",
        "mode": "chat",
        "use_web_search": False,
        "history": [],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["response"]) > 0
    assert data["mode"] == "chat"


@pytest.mark.skipif(not _has_chat_provider, reason="No chat provider configured")
def test_chat_message_returns_suggestions():
    response = client.post("/api/v1/chat/message", json={
        "message": "Explain gradient descent.",
        "mode": "chat",
        "use_web_search": False,
        "history": [],
    })
    data = response.json()
    assert data["success"] is True
    assert isinstance(data.get("suggestions", []), list)


@pytest.mark.skipif(not _has_chat_provider, reason="No chat provider configured")
def test_chat_explain_mode():
    response = client.post("/api/v1/chat/message", json={
        "message": "Explain backpropagation",
        "mode": "explain",
        "explain_level": "beginner",
        "explain_style": "technical",
        "use_web_search": False,
        "history": [],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["response"]) > 0


# ---------------------------------------------------------------------------
# Session management (no LLM needed)
# ---------------------------------------------------------------------------

def test_list_sessions_returns_200():
    response = client.get("/api/v1/chat/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    assert isinstance(data["sessions"], list)


def test_get_nonexistent_session_returns_404():
    response = client.get("/api/v1/chat/sessions/nonexistent-session-id")
    assert response.status_code == 404


def test_delete_nonexistent_session_returns_404():
    response = client.delete("/api/v1/chat/sessions/nonexistent-session-id")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Topics and documents endpoints (no LLM needed)
# ---------------------------------------------------------------------------

def test_get_topics_returns_200():
    response = client.get("/api/v1/topics")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "topics" in data


def test_list_documents_returns_200():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "documents" in data


def test_get_nonexistent_document_returns_404():
    response = client.get("/api/v1/documents/nonexistent-doc-id-xyz")
    assert response.status_code == 404
