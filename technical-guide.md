# EduNotes Architecture

This document covers non-obvious design decisions that are difficult to infer from the code alone.

---

## System Overview

EduNotes is a multi-agent study assistant. A user submits a query (topic, URL, or raw text) through the Streamlit UI or directly to the FastAPI backend. The Orchestrator routes the query to the appropriate pipeline, coordinates agents, and returns structured markdown notes.

```
User → FastAPI → Orchestrator
                    ├── TOPIC  → WebSearchAgent → ContentAgent → NoteMaker
                    ├── URL    → ScraperAgent   → ContentAgent → NoteMaker
                    └── TEXT   →                  ContentAgent → NoteMaker

                 RetrieverAgent (ChromaDB KB) — consulted before web search in auto mode
                 ChatAgent — separate conversational pipeline, not part of note generation
```

---

## Dual LLM Client Design

There are two separate LLM clients that must remain separate:

| Client | File | Used By | Providers |
|---|---|---|---|
| `LLMClient` | `src/utils/llm_client.py` | Summarization pipeline | Groq (primary) → HuggingFace (fallback) |
| `GeminiClient` | `src/utils/gemini_client.py` | Chat pipeline | Gemini (primary) → Cerebras (fallback) |

**Why they are separate:** Both pipelines make LLM calls during the same user session. Groq's free tier has strict per-minute rate limits. If the chat agent shared the same Groq client as the summarization pipeline, simultaneous or back-to-back requests would exhaust the rate limit for both features. Keeping them on different providers eliminates this conflict entirely.

The chat pipeline's fallback chain: `Gemini → Cerebras → Groq light model`. Groq appears at the end of the chat fallback chain only — never as a primary for chat.

---

## ContentAgent — 4-Stage Pipeline

`ContentAgent` wraps `SummarizerAgent` with intelligence. It is the processing core for all content types (web pages, scraped text, raw input).

**Stage 1 — Content Type Analysis (LLM):** Classifies content into: `academic`, `tutorial`, `research`, `reference`, or `general`. Uses Groq with a lightweight prompt against the first 1,000 characters. Result is cached for 24 hours by MD5 hash of the content preview. Falls back to `general` if no LLM is available.

**Stage 2 — Strategy Selection (rule-based, no LLM):** Maps the content type to a set of preservation instructions injected into the summarizer prompt. For example, `academic` content gets instructions to preserve LaTeX equations and markdown tables verbatim; `tutorial` content preserves numbered steps and code blocks. `general` type gets no extra instructions.

**Stage 3 — Content Processing (SummarizerAgent):** Delegates to `SummarizerAgent`, passing both the content and any extra instructions from Stage 2.

**Stage 4 — Self-Evaluation (LLM):** After summarization, asks the LLM whether any critical topics from the original content are missing from the summary. If gaps are found, `needs_more_info=True` and `gap_queries` (a list of missing topics) are set on the result. The Orchestrator then calls `_fill_content_gaps()`, which runs additional WebSearchAgent queries and re-summarizes the enriched content.

**Bypass:** Flashcard and quiz modes skip Stages 1, 2, and 4 — they delegate directly to SummarizerAgent because gap detection is not meaningful for those output formats. Authoritative content (URL scrapes, PDF, direct text input) also skips Stage 4 because the full original content is already present; gap detection would produce false positives by comparing summarized output against content that was intentionally condensed.

---

## WebSearchAgent — 5-Stage Pipeline

`WebSearchAgent` is called for TOPIC queries and for gap resolution. It uses Groq as the reasoning brain and DuckDuckGo + ScraperAgent as tools.

**Stage 1 — Query Generation (LLM):** Reformulates the user's topic into 2–3 effective search queries. This handles ambiguous short inputs (e.g., "CNN" → "convolutional neural network deep learning").

**Stage 2 — Web Search (DuckDuckGo):** Executes the generated queries. Falls back to a secondary search provider if DuckDuckGo fails.

**Stage 3 — Result Evaluation (LLM):** Ranks the returned URLs for educational relevance, filtering out low-quality or irrelevant results before scraping.

**Stage 4 — Content Extraction (ScraperAgent):** Scrapes the top-ranked URLs. Uses `newspaper3k` as primary, `BeautifulSoup` as fallback.

**Stage 5 — Quality Assessment (LLM):** Validates that the scraped content is substantive and educationally useful before passing it to ContentAgent.

---

## Key Behavioral Flags

### `save_to_kb` (default: `False`)
Controls whether generated notes are persisted to ChromaDB after processing. Defaults to `False` because bulk saves on memory-constrained deployments (e.g., 512MB free-tier instances) can cause OOM errors when the embedding model processes large documents. Users opt in explicitly via the UI toggle or the API field.

### `search_mode` routing
Controls where TOPIC queries look for content:

| Mode | Behavior |
|---|---|
| `auto` | KB first; if insufficient results, falls back to web search with a note in the output |
| `kb_only` | Knowledge base only; no web calls |
| `web_search` | Web only; KB is skipped entirely |
| `both` | KB and web searched in parallel; results merged before summarization |

### `use_web_search` (chat pipeline only)
When `True`, the ChatAgent performs a web search to ground its response before generating. When `False`, it answers from its LLM knowledge and any retrieved KB context.

### Gap Resolution
After ContentAgent Stage 4, if `needs_more_info=True`, the Orchestrator calls `_fill_content_gaps()`, which runs up to 2 additional WebSearchAgent queries (one per identified gap topic). The supplementary content is appended to the original content and re-processed by ContentAgent with `skip_evaluation=True` to avoid a recursive evaluation loop.

---

## Embeddings — ONNX Runtime

The vector store uses ChromaDB's built-in `ONNXMiniLM_L6_V2` embedding function instead of loading the model through `sentence-transformers` or `torch`. This means:

- No `torch` or `transformers` install required — eliminates ~2GB of dependencies.
- The same MiniLM-L6-v2 model weights run via ONNX Runtime, which is already a ChromaDB dependency.
- Cold-start embedding time is faster since the ONNX model is already optimized for inference.

The trade-off is that custom embedding models cannot be swapped in without replacing the ChromaDB embedding function. If a different embedding model is needed in the future, use `chromadb.utils.embedding_functions` to wrap it.

---

## Provider Fallback Chains

**Summarization pipeline** (`LLMClient`):
```
Groq (llama-3.3-70b) → Groq light model (rate limit) → HuggingFace
```

**Chat pipeline** (`ChatLLMClient` / `GeminiClient`):
```
Gemini (gemini-2.5-flash) → Cerebras → Groq light model
```

Both chains are initialized at startup. If the primary provider's API key is missing or the client fails to initialize, the next provider in the chain is attempted automatically. `is_available()` on each client returns `False` if no provider initialized successfully, allowing callers to degrade gracefully (e.g., ContentAgent skips Stage 1 and 4 classification when the LLM is unavailable).
