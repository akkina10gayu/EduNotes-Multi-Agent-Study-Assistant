"""
Unit tests for pure utility functions in text_utils.py.
No external calls, no LLM, no database. These run instantly.
"""
import pytest

from src.utils.text_utils import (
    clean_text,
    chunk_text,
    _sanitize_llm_summary,
    format_as_markdown,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

def test_clean_text_collapses_whitespace():
    assert clean_text("hello   world") == "hello world"


def test_clean_text_strips_leading_trailing():
    assert clean_text("  hello  ") == "hello"


def test_clean_text_removes_at_symbol():
    result = clean_text("hello@world")
    assert "@" not in result


def test_clean_text_keeps_punctuation():
    result = clean_text("Hello, world! Is this working?")
    assert "," in result
    assert "!" in result
    assert "?" in result


def test_clean_text_empty_string():
    assert clean_text("") == ""


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_produces_multiple_chunks():
    text = " ".join(["word"] * 1000)
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1


def test_chunk_text_empty_input():
    assert chunk_text("") == []


def test_chunk_text_short_text_is_single_chunk():
    text = "hello world"
    chunks = chunk_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_overlap_produces_shared_words():
    # With overlap > 0, consecutive chunks should share some words
    text = " ".join([str(i) for i in range(200)])
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 2
    last_words_of_first = set(chunks[0].split()[-5:])
    first_words_of_second = set(chunks[1].split()[:5])
    assert last_words_of_first & first_words_of_second  # some overlap exists


# ---------------------------------------------------------------------------
# _sanitize_llm_summary
# ---------------------------------------------------------------------------

def test_sanitize_removes_long_code_block():
    lines = "\n".join([f"line_{i} = {i}" for i in range(15)])
    text = f"Some intro\n```python\n{lines}\n```\nSome outro"
    result = _sanitize_llm_summary(text)
    assert "```" not in result
    assert "Some intro" in result
    assert "Some outro" in result


def test_sanitize_keeps_short_code_block():
    text = "Here is an example:\n```python\nprint('hello')\n```\nEnd."
    result = _sanitize_llm_summary(text)
    assert "```" in result


def test_sanitize_removes_undefined_on_own_line():
    text = "Content here\nundefined\nMore content"
    result = _sanitize_llm_summary(text)
    assert "undefined" not in result


def test_sanitize_keeps_undefined_inside_sentence():
    # "undefined" inside a sentence should not be stripped
    text = "The variable is undefined in this context."
    result = _sanitize_llm_summary(text)
    assert "undefined" in result


def test_sanitize_collapses_excessive_blank_lines():
    text = "Line 1\n\n\n\n\nLine 2"
    result = _sanitize_llm_summary(text)
    assert "\n\n\n" not in result


def test_sanitize_empty_string():
    assert _sanitize_llm_summary("") == ""


def test_sanitize_no_code_blocks_unchanged():
    text = "Plain text with no code blocks at all."
    result = _sanitize_llm_summary(text)
    assert result == text


# ---------------------------------------------------------------------------
# format_as_markdown
# ---------------------------------------------------------------------------

def _make_content(summary: str, mode: str, sources=None):
    content = {
        "summary": summary,
        "metadata": {
            "date": "2025-01-01 00:00:00",
            "topic": "Test Topic",
            "summarization_mode": mode,
        },
    }
    if sources:
        content["sources"] = sources
    return content


def test_format_markdown_includes_title():
    md = format_as_markdown("My Title", _make_content("Text here.", "paragraph_summary"))
    assert "### My Title" in md


def test_format_markdown_paragraph_summary_has_overview_header():
    md = format_as_markdown("Title", _make_content("Some text.", "paragraph_summary"))
    assert "#### Overview" in md


def test_format_markdown_important_points_no_overview_header():
    md = format_as_markdown("Title", _make_content("1. Point one\n2. Point two", "important_points"))
    assert "#### Overview" not in md


def test_format_markdown_key_highlights_no_overview_header():
    md = format_as_markdown("Title", _make_content("• Term: definition", "key_highlights"))
    assert "#### Overview" not in md


def test_format_markdown_includes_valid_http_source():
    sources = [{"title": "Valid Source", "url": "https://example.com"}]
    md = format_as_markdown("Title", _make_content("Text.", "paragraph_summary", sources))
    assert "https://example.com" in md
    assert "Valid Source" in md


def test_format_markdown_excludes_ftp_source():
    sources = [
        {"title": "FTP", "url": "ftp://files.example.com"},
        {"title": "Valid", "url": "https://example.com"},
    ]
    md = format_as_markdown("Title", _make_content("Text.", "paragraph_summary", sources))
    assert "ftp://files.example.com" not in md
    assert "https://example.com" in md


def test_format_markdown_excludes_empty_url_source():
    sources = [{"title": "No URL", "url": ""}]
    md = format_as_markdown("Title", _make_content("Text.", "paragraph_summary", sources))
    assert "#### Sources" not in md


def test_format_markdown_includes_metadata_date():
    md = format_as_markdown("Title", _make_content("Text.", "paragraph_summary"))
    assert "2025-01-01" in md


def test_format_markdown_balanced_code_fences():
    # If the summary has an unclosed code fence, format_as_markdown should close it
    # before appending sources/metadata so those sections render correctly
    summary_with_open_fence = "Some text\n```python\ncode here"
    md = format_as_markdown("Title", _make_content(summary_with_open_fence, "paragraph_summary"))
    fence_count = sum(1 for ln in md.split("\n") if ln.lstrip().startswith("```"))
    assert fence_count % 2 == 0
