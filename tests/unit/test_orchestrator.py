"""
Unit tests for Orchestrator routing logic and pure helper methods.
No external API calls are made in these tests.
"""
import inspect
import pytest

from src.agents.orchestrator import Orchestrator, QueryType

orchestrator = Orchestrator()


# ---------------------------------------------------------------------------
# detect_query_type
# ---------------------------------------------------------------------------

def test_detect_http_url():
    assert orchestrator.detect_query_type("http://example.com") == QueryType.URL


def test_detect_https_url():
    assert orchestrator.detect_query_type("https://example.com/article/123") == QueryType.URL


def test_detect_www_url():
    assert orchestrator.detect_query_type("www.example.com") == QueryType.URL


def test_detect_topic_short_string():
    assert orchestrator.detect_query_type("machine learning") == QueryType.TOPIC


def test_detect_topic_at_boundary():
    # Exactly 500 chars is still TOPIC; TEXT requires > 500
    assert orchestrator.detect_query_type("a" * 500) == QueryType.TOPIC


def test_detect_text_over_500_chars():
    assert orchestrator.detect_query_type("a" * 501) == QueryType.TEXT


def test_detect_text_long_paragraph():
    long_text = "This is a sentence. " * 30  # ~600 chars
    assert orchestrator.detect_query_type(long_text) == QueryType.TEXT


# ---------------------------------------------------------------------------
# process() signature — save_to_kb must default to False
# ---------------------------------------------------------------------------

def test_save_to_kb_defaults_false():
    sig = inspect.signature(orchestrator.process)
    assert sig.parameters["save_to_kb"].default is False


def test_process_topic_query_save_to_kb_defaults_false():
    sig = inspect.signature(orchestrator.process_topic_query)
    assert sig.parameters["save_to_kb"].default is False


def test_process_url_query_save_to_kb_defaults_false():
    sig = inspect.signature(orchestrator.process_url_query)
    assert sig.parameters["save_to_kb"].default is False


# ---------------------------------------------------------------------------
# _deduplicate_sources
# ---------------------------------------------------------------------------

def test_deduplicate_removes_duplicate_urls():
    results = [
        {"metadata": {"url": "https://example.com", "title": "Example"}},
        {"metadata": {"url": "https://example.com", "title": "Example duplicate"}},
        {"metadata": {"url": "https://other.com", "title": "Other"}},
    ]
    sources = orchestrator._deduplicate_sources(results)
    assert len(sources) == 2
    urls = [s["url"] for s in sources]
    assert len(set(urls)) == len(urls)


def test_deduplicate_excludes_non_http_urls():
    results = [
        {"metadata": {"url": "ftp://files.example.com", "title": "FTP"}},
        {"metadata": {"url": "", "title": "No URL"}},
        {"metadata": {"url": "https://valid.com", "title": "Valid"}},
    ]
    sources = orchestrator._deduplicate_sources(results)
    assert len(sources) == 1
    assert sources[0]["url"] == "https://valid.com"


def test_deduplicate_empty_results():
    assert orchestrator._deduplicate_sources([]) == []


def test_deduplicate_preserves_title():
    results = [{"metadata": {"url": "https://example.com", "title": "My Title"}}]
    sources = orchestrator._deduplicate_sources(results)
    assert sources[0]["title"] == "My Title"


# ---------------------------------------------------------------------------
# _balance_code_fences (static method)
# ---------------------------------------------------------------------------

def test_balance_fences_already_balanced():
    text = "```python\nprint('hello')\n```"
    assert Orchestrator._balance_code_fences(text) == text


def test_balance_fences_odd_count_appends_closing():
    text = "```python\nprint('hello')"
    result = Orchestrator._balance_code_fences(text)
    assert result.endswith("```\n")


def test_balance_fences_empty_string():
    assert Orchestrator._balance_code_fences("") == ""


def test_balance_fences_no_fences():
    text = "Just plain text with no code blocks."
    assert Orchestrator._balance_code_fences(text) == text


def test_balance_fences_multiple_balanced_blocks():
    text = "```python\ncode1\n```\nSome text\n```bash\ncode2\n```"
    assert Orchestrator._balance_code_fences(text) == text


# ---------------------------------------------------------------------------
# TITLE_SUFFIX_MAP
# ---------------------------------------------------------------------------

def test_title_suffix_map_has_all_modes():
    expected = {"paragraph_summary", "important_points", "key_highlights"}
    assert set(orchestrator.TITLE_SUFFIX_MAP.keys()) == expected


def test_title_suffix_paragraph_summary_is_empty():
    # paragraph_summary adds no suffix — title is just the topic
    assert orchestrator.TITLE_SUFFIX_MAP["paragraph_summary"] == ""


def test_title_suffix_important_points():
    assert "Important Points" in orchestrator.TITLE_SUFFIX_MAP["important_points"]


def test_title_suffix_key_highlights():
    assert "Key Highlights" in orchestrator.TITLE_SUFFIX_MAP["key_highlights"]
