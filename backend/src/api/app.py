"""
FastAPI application for EduNotes
"""
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from datetime import datetime
import asyncio
import json
import re
import base64
import time as _time

from src.models.schemas import (
    GenerateNotesRequest, GenerateNotesResponse,
    UpdateKBRequest, UpdateKBResponse,
    SearchKBRequest, SearchKBResponse,
    StatsResponse, HealthResponse
)
from src.agents.orchestrator import Orchestrator
from src.db import document_store
from src.db.progress_store import record_activity
from src.api.auth import get_current_user
from src.utils.logger import get_logger
from src.utils.pdf_processor import get_pdf_processor
from src.api.study_routes import router as study_router
from config import settings

logger = get_logger(__name__)


def _clean_vision_desc(desc: str) -> str:
    """Remove absent-category notices, HTML tags, and excessive whitespace from vision output."""
    if not desc:
        return ''
    # Replace HTML <br> tags with newlines
    desc = re.sub(r'<br\s*/?>', '\n', desc)
    # Remove remaining HTML tags
    desc = re.sub(r'<[^>]+>', '', desc)
    # Remove lines about absent content and their headings
    lines = desc.split('\n')
    cleaned = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # Line says "No X present/found/visible/etc"
        if re.search(r'\bno\b.{0,40}\b(present|found|detected|visible|shown|appear)', stripped, re.IGNORECASE):
            # Drop preceding heading if it belongs to this empty section
            if cleaned and re.match(r'^#{1,6}\s', cleaned[-1].strip()):
                cleaned.pop()
            i += 1
            continue
        # Heading followed immediately by a "no X" line → skip both
        if re.match(r'^#{1,6}\s', stripped) and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if re.search(r'\bno\b.{0,40}\b(present|found|detected|visible|shown|appear)', next_stripped, re.IGNORECASE):
                i += 2
                continue
        cleaned.append(lines[i])
        i += 1
    desc = '\n'.join(cleaned)
    # Collapse 3+ blank lines to 2
    desc = re.sub(r'\n{3,}', '\n\n', desc)
    return desc.strip()


# Initialize FastAPI app
app = FastAPI(
    title="EduNotes API",
    description="Multi-Agent Study Assistant API with Study Features (Flashcards, Quizzes, Progress Tracking)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression middleware (Phase 5 optimization)
# Compresses responses larger than 500 bytes for faster transfer
app.add_middleware(GZipMiddleware, minimum_size=500)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATE_LIMIT_PER_HOUR}/hour"]
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include study routes
app.include_router(study_router)

# Initialize components
orchestrator = Orchestrator()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting EduNotes API...")
    logger.info(f"API running on {settings.API_HOST}:{settings.API_PORT}")

    # Validate API keys
    if settings.LLM_PROVIDER == "groq":
        if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_key_here":
            logger.warning("="*80)
            logger.warning("⚠️  GROQ API KEY NOT CONFIGURED!")
            logger.warning("="*80)
            logger.warning("The system will use local models (slower performance).")
            logger.warning("To use the faster Groq API:")
            logger.warning("1. Get a free API key at: https://console.groq.com")
            logger.warning("2. Add it to your .env file: GROQ_API_KEY=your_key_here")
            logger.warning("="*80)
        else:
            logger.info("✅ Groq API key configured")

    elif settings.LLM_PROVIDER == "huggingface":
        if not settings.HF_TOKEN or settings.HF_TOKEN == "your_hf_token_here":
            logger.warning("="*80)
            logger.warning("⚠️  HUGGINGFACE TOKEN NOT CONFIGURED!")
            logger.warning("="*80)
            logger.warning("The system will use local models.")
            logger.warning("To use HuggingFace API:")
            logger.warning("1. Get a token at: https://huggingface.co/settings/tokens")
            logger.warning("2. Add it to your .env file: HF_TOKEN=your_token_here")
            logger.warning("="*80)
        else:
            logger.info("✅ HuggingFace token configured")

    else:
        logger.info("Using local models (offline mode)")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down EduNotes API...")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to EduNotes API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/api/v1/generate-notes",
          response_model=GenerateNotesResponse,
          tags=["Notes"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def generate_notes(request: Request, body: GenerateNotesRequest, user_id: str = Depends(get_current_user)):
    """Generate study notes from query"""
    try:
        # Input validation
        query = body.query.strip()

        # Check if query is empty
        if not query:
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty. Please provide a topic, URL, or text to generate notes."
            )

        # Check minimum length
        if len(query) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query is too short. Please provide at least 3 characters."
            )

        # Check maximum length (100KB)
        if len(query) > 100000:
            raise HTTPException(
                status_code=400,
                detail="Query is too long. Maximum length is 100,000 characters. Please provide shorter content."
            )

        # Validate URL format if it looks like a URL
        if query.startswith(('http://', 'https://', 'www.')):
            import re
            url_pattern = re.compile(
                r'^(?:http|https)://(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
                r'localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
                r'(?::\d+)?(?:/?|[/?]\S+)$', re.IGNORECASE
            )
            normalized_url = query if query.startswith('http') else f'https://{query}'
            if not url_pattern.match(normalized_url):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid URL format. Please provide a valid URL (e.g., https://example.com)"
                )

        logger.info(f"Generating notes for query: {query[:50]}... Mode: {body.summarization_mode}, Length: {body.summary_length}")

        # Detect PDF URLs — download and process through PDF pipeline instead of scraping
        if query.startswith(('http://', 'https://')) and (
            query.lower().endswith('.pdf') or '/pdf/' in query.lower()
        ):
            logger.info(f"Detected PDF URL: {query}, downloading for PDF extraction")
            _pdf_url_t0 = _time.time()
            _pdf_processed = False
            try:
                import aiohttp
                async with aiohttp.ClientSession() as http_session:
                    async with http_session.get(
                        query,
                        timeout=aiohttp.ClientTimeout(total=30),
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0',
                            'Accept': 'application/pdf,*/*'
                        },
                        allow_redirects=True
                    ) as resp:
                        if resp.status != 200:
                            raise Exception(f"HTTP {resp.status}")
                        pdf_content = await resp.read()

                # Verify it's actually a PDF (magic bytes), not an HTML page
                if not pdf_content[:5].startswith(b'%PDF-'):
                    logger.info("URL returned non-PDF content (likely HTML), falling back to URL scraping")
                    raise Exception("Response is not a PDF file")

                file_size_mb = len(pdf_content) / (1024 * 1024)
                logger.info(f"Downloaded PDF: {file_size_mb:.2f}MB")
                if file_size_mb > 20:
                    raise HTTPException(
                        status_code=400,
                        detail=f"PDF too large ({file_size_mb:.1f}MB). Maximum is 20MB."
                    )

                pdf_processor = get_pdf_processor()
                figures_for_ui = []
                if body.research_mode:
                    research_data = pdf_processor.extract_for_research(pdf_content)
                    extracted_text = research_data.get('text', '')

                    # Vision analysis for figure pages (same as process_pdf)
                    figure_pages = research_data.get('figure_pages', [])
                    if figure_pages:
                        try:
                            from src.utils.llm_client import get_llm_client
                            llm_client = get_llm_client()

                            vision_prompt = (
                                "Analyze this page. Describe ONLY what you actually see.\n\n"
                                "Rules:\n"
                                "- NEVER use HTML tags (no <br>, <b>, <table> etc). Use markdown only.\n"
                                "- NEVER mention categories that have nothing on this page.\n"
                                "- NEVER write \"No X present\" or \"No X found\" for any category.\n"
                                "- If only a figure exists, describe only the figure. Skip everything else silently.\n\n"
                                "Always start with the figure/table identifier from the caption "
                                "(e.g. 'Figure 1:', 'Table 3:', 'Fig. 2:') if one is visible.\n"
                                "For figures/diagrams: What it shows, key trends, data points.\n"
                                "For equations: LaTeX inline (e.g. $E = mc^2$).\n"
                                "For tables: Proper markdown table with | and --- separators. "
                                "One value per cell, one row per line. Never combine rows with <br>.\n"
                                "For charts/plots: Axes labels, trends, key values.\n\n"
                                "Be concise and technical. No preamble, no step numbers."
                            )

                            for fig_page in figure_pages:
                                desc = llm_client.describe_image(
                                    fig_page['image_base64'], vision_prompt
                                )
                                desc = _clean_vision_desc(desc) if desc else ''
                                if desc:
                                    figures_for_ui.append({
                                        "page": fig_page['page_num'],
                                        "description": desc,
                                        "image_b64": fig_page['image_base64']
                                    })
                                    logger.info(f"PDF URL vision analysis complete for page {fig_page['page_num']}")

                            if figures_for_ui:
                                logger.info(f"PDF URL: prepared {len(figures_for_ui)} figure entries for UI expander")

                        except Exception as e:
                            logger.warning(f"PDF URL vision analysis failed, proceeding with text only: {e}")
                else:
                    extracted_text = pdf_processor.extract_text_from_bytes(pdf_content)

                if not extracted_text or len(extracted_text) < 50:
                    raise Exception("PDF text extraction returned insufficient content")

                logger.info(f"Extracted {len(extracted_text)} chars from PDF URL")

                # Clean HTML artifacts from PDF extraction
                extracted_text = re.sub(r'<br\s*/?>', ', ', extracted_text)

                result = await orchestrator.process(
                    extracted_text,
                    summarization_mode=body.summarization_mode,
                    output_length=body.summary_length,
                    research_mode=body.research_mode or False,
                    user_id=user_id
                )

                if not result.get('success', False):
                    result.setdefault('query_type', 'url')
                    result.setdefault('query', query[:100])
                    result.setdefault('notes', '')
                    result.setdefault('sources_used', 0)
                    result.setdefault('from_kb', False)

                # Clean any remaining HTML artifacts from output notes
                if result.get('notes'):
                    result['notes'] = re.sub(r'<br\s*/?>', ', ', result['notes'])

                # Attach structured vision data for UI expander
                if figures_for_ui and result.get('success'):
                    result['vision_data'] = json.dumps(figures_for_ui)
                    logger.info(f"PDF URL: attached {len(figures_for_ui)} figure entries as vision_data")

                if result.get('success', False):
                    try:
                        record_activity(user_id, "note_generated", f"PDF: {query.split('/')[-1][:30]}", {})
                    except Exception:
                        pass

                _pdf_processed = True
                _elapsed = _time.time() - _pdf_url_t0
                logger.info(f"PDF URL pipeline completed in {_elapsed:.1f}s — success={result.get('success')}, notes_len={len(result.get('notes', ''))}")
                return GenerateNotesResponse(**result)

            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"PDF URL processing failed: {e}, falling back to URL scraping")
                # Fall through to normal URL processing below

            # If PDF download/extraction failed, try the abstract page for arxiv URLs
            if not _pdf_processed and 'arxiv.org/pdf/' in query:
                abs_url = query.replace('/pdf/', '/abs/').split('.pdf')[0]
                logger.info(f"Retrying with arxiv abstract page: {abs_url}")
                query = abs_url

        # Process query through orchestrator
        result = await orchestrator.process(
            query,
            summarization_mode=body.summarization_mode,
            output_length=body.summary_length,
            search_mode=body.search_mode or "auto",
            research_mode=body.research_mode or False,
            user_id=user_id
        )

        # Ensure error responses have all required fields for the response model
        if not result.get('success', False):
            result.setdefault('query_type', 'unknown')
            result.setdefault('query', query[:100])
            result.setdefault('notes', '')
            result.setdefault('sources_used', 0)
            result.setdefault('from_kb', False)

        # Record activity for progress tracking (updates streak)
        if result.get('success', False):
            try:
                # Use query type as topic, or first 50 chars of query
                topic = result.get('query_type', 'general')
                if topic == 'text':
                    topic = query[:50].strip() if len(query) > 50 else query.strip()
                record_activity(user_id, "note_generated", topic, {})
                logger.debug(f"Recorded note generation activity for topic: {topic}")
            except Exception as e:
                logger.warning(f"Failed to record activity: {e}")
                # Don't fail the request if activity recording fails

        return GenerateNotesResponse(**result)

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error generating notes: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/update-kb",
          response_model=UpdateKBResponse,
          tags=["Knowledge Base"])
@limiter.limit("10/minute")
async def update_knowledge_base(request: Request, body: UpdateKBRequest, user_id: str = Depends(get_current_user)):
    """Update knowledge base with new documents"""
    try:
        # Input validation
        if not body.documents:
            raise HTTPException(
                status_code=400,
                detail="No documents provided. Please include at least one document."
            )

        if len(body.documents) > 100:
            raise HTTPException(
                status_code=400,
                detail="Too many documents. Maximum 100 documents per request."
            )

        # Validate each document
        for idx, doc in enumerate(body.documents):
            if not doc.get('content', '').strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {idx + 1} has empty content. All documents must have content."
                )
            if len(doc.get('content', '')) > 50000:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {idx + 1} is too long. Maximum 50,000 characters per document."
                )

        logger.info(f"Updating KB with {len(body.documents)} documents")

        # Process documents — loop and add each via document_store
        documents_added = 0
        for doc in body.documents:
            try:
                document_store.add_document(
                    user_id,
                    title=doc.get('title', ''),
                    topic=doc.get('topic', ''),
                    content=doc.get('content', ''),
                    source=doc.get('source', ''),
                    url=doc.get('url', '')
                )
                documents_added += 1
            except Exception as e:
                logger.warning(f"Failed to add document: {e}")

        if documents_added > 0:
            return UpdateKBResponse(
                success=True,
                documents_added=documents_added
            )
        else:
            return UpdateKBResponse(
                success=False,
                documents_added=0,
                error="No documents were processed successfully"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating KB: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/search-kb",
          response_model=SearchKBResponse,
          tags=["Knowledge Base"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def search_knowledge_base(request: Request, body: SearchKBRequest, user_id: str = Depends(get_current_user)):
    """Search the knowledge base"""
    try:
        logger.info(f"Searching KB for: {body.query[:50]}...")

        results = document_store.search(
            user_id,
            query=body.query,
            k=body.k,
            threshold=body.threshold
        )

        return SearchKBResponse(
            success=True,
            query=body.query,
            results=results,
            count=len(results)
        )

    except Exception as e:
        logger.error(f"Error searching KB: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/process-pdf",
          response_model=GenerateNotesResponse,
          tags=["Notes"])
@limiter.limit("20/minute")
async def process_pdf(
    request: Request,
    file: UploadFile = File(None),
    summarization_mode: str = Form("paragraph_summary"),
    output_length: str = Form("auto"),
    cached_text: str = Form(None),
    cached_filename: str = Form(None),
    research_mode: bool = Form(False),
    user_id: str = Depends(get_current_user)
):
    """Process PDF file and generate study notes.

    Supports two modes:
    1. Normal: Upload PDF file for extraction and processing
    2. Cached: Provide pre-extracted text (cached_text) to skip extraction (Phase 5 optimization)
    """
    try:
        extracted_text = None
        filename = None
        figures_for_ui = []  # Structured figure data for UI expander

        # Phase 5: Check if using cached text (skip extraction)
        if cached_text:
            extracted_text = cached_text
            filename = cached_filename or "cached_pdf"
            logger.info(f"Using cached text for PDF: {filename} ({len(extracted_text)} chars)")
        else:
            # Normal mode: Process uploaded PDF file
            if not file or not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="No PDF file provided. Please upload a PDF file."
                )

            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Please upload a PDF file."
                )

            # Check file size (max 10MB)
            content = await file.read()
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large ({file_size_mb:.1f}MB). Maximum size is 10MB."
                )

            filename = file.filename
            logger.info(f"Processing PDF: {filename} ({file_size_mb:.2f}MB), research_mode={research_mode}")

            pdf_processor = get_pdf_processor()

            if research_mode:
                # Research Mode: enhanced extraction + vision analysis
                research_data = pdf_processor.extract_for_research(content)
                extracted_text = research_data.get('text', '')

                # Vision analysis for figure pages
                figure_pages = research_data.get('figure_pages', [])
                if figure_pages:
                    try:
                        from src.utils.llm_client import get_llm_client
                        llm_client = get_llm_client()

                        vision_prompt = (
                            "Analyze this page. Describe ONLY what you actually see.\n\n"
                            "Rules:\n"
                            "- NEVER use HTML tags (no <br>, <b>, <table> etc). Use markdown only.\n"
                            "- NEVER mention categories that have nothing on this page.\n"
                            "- NEVER write \"No X present\" or \"No X found\" for any category.\n"
                            "- If only a figure exists, describe only the figure. Skip everything else silently.\n\n"
                            "Always start with the figure/table identifier from the caption "
                            "(e.g. 'Figure 1:', 'Table 3:', 'Fig. 2:') if one is visible.\n"
                            "For figures/diagrams: What it shows, key trends, data points.\n"
                            "For equations: LaTeX inline (e.g. $E = mc^2$).\n"
                            "For tables: Proper markdown table with | and --- separators. "
                            "One value per cell, one row per line. Never combine rows with <br>.\n"
                            "For charts/plots: Axes labels, trends, key values.\n\n"
                            "Be concise and technical. No preamble, no step numbers."
                        )

                        for fig_page in figure_pages:
                            desc = llm_client.describe_image(
                                fig_page['image_base64'], vision_prompt
                            )
                            desc = _clean_vision_desc(desc) if desc else ''
                            if desc:
                                figures_for_ui.append({
                                    "page": fig_page['page_num'],
                                    "description": desc,
                                    "image_b64": fig_page['image_base64']
                                })
                                logger.info(f"Vision analysis complete for page {fig_page['page_num']}")

                        if figures_for_ui:
                            logger.info(f"Prepared {len(figures_for_ui)} figure entries for UI expander")

                    except Exception as e:
                        logger.warning(f"Vision analysis failed, proceeding with text only: {e}")
            else:
                # Standard extraction
                extracted_text = pdf_processor.extract_text_from_bytes(content)

        # Validate extracted text
        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. The PDF may be image-based or encrypted."
            )

        if len(extracted_text) < 50:
            raise HTTPException(
                status_code=400,
                detail="Extracted text is too short. The PDF may not contain enough readable content."
            )

        # Clean HTML artifacts from PDF extraction (pymupdf4llm puts <br> in table cells)
        extracted_text = re.sub(r'<br\s*/?>', ', ', extracted_text)

        logger.info(f"Processing {len(extracted_text)} characters with summarization_mode='{summarization_mode}', output_length='{output_length}', research_mode={research_mode}")

        # Process through orchestrator as text query
        result = await orchestrator.process(
            extracted_text,
            summarization_mode=summarization_mode,
            output_length=output_length,
            research_mode=research_mode,
            user_id=user_id
        )

        logger.info(f"PDF processing completed - Success: {result.get('success', False)}")

        # Ensure error responses have all required fields for the response model
        if not result.get('success', False):
            result.setdefault('query_type', 'text')
            result.setdefault('query', f"PDF: {filename}"[:100])
            result.setdefault('notes', '')
            result.setdefault('sources_used', 0)
            result.setdefault('from_kb', False)

        # Clean any remaining HTML artifacts from output notes
        if result.get('notes'):
            result['notes'] = re.sub(r'<br\s*/?>', ', ', result['notes'])

        # Attach structured vision data for UI expander (not mixed into notes)
        if figures_for_ui and result.get('success'):
            result['vision_data'] = json.dumps(figures_for_ui)
            logger.info(f"Attached {len(figures_for_ui)} figure entries as vision_data")

        # Add PDF filename and extracted text to result (Phase 5: for caching)
        if result.get('success'):
            result['source_file'] = filename
            result['extracted_text'] = extracted_text  # Phase 5: Return for UI caching
            # Record activity for progress tracking (updates streak)
            try:
                topic = f"PDF: {filename[:30]}"
                record_activity(user_id, "note_generated", topic, {})
                logger.debug(f"Recorded PDF note generation activity")
            except Exception as e:
                logger.warning(f"Failed to record activity: {e}")

        return GenerateNotesResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/v1/topics",
         tags=["Knowledge Base"])
async def get_topics(user_id: str = Depends(get_current_user)):
    """Get available topics from knowledge base"""
    try:
        # Get unique topics from document store
        topics = document_store.get_unique_topics(user_id)

        return {
            "success": True,
            "topics": topics[:20],  # Limit to top 20 topics
            "total_topics": len(topics)
        }

    except Exception as e:
        logger.error(f"Error getting topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents",
         tags=["Knowledge Base"])
async def list_documents(keyword: str = None, user_id: str = Depends(get_current_user)):
    """List all documents in the knowledge base, optionally filtered by keyword"""
    try:
        if keyword and keyword.strip():
            documents = document_store.search_documents_semantic(user_id, keyword.strip())
        else:
            documents = document_store.list_documents(user_id)

        return {
            "success": True,
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/search",
         tags=["Knowledge Base"])
async def search_documents_semantic(query: str, k: int = 20, user_id: str = Depends(get_current_user)):
    """Semantic search for full documents via vector DB chunk matching.
    Searches chunks by meaning, then maps matching chunks back to their
    parent documents using title metadata.
    """
    try:
        if not query or not query.strip():
            return {"success": True, "documents": [], "count": 0}

        # Search documents semantically via document_store
        documents = document_store.search_documents_semantic(user_id, query.strip())

        return {
            "success": True,
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        logger.error(f"Error in semantic document search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{doc_id}",
         tags=["Knowledge Base"])
async def get_document(doc_id: str, user_id: str = Depends(get_current_user)):
    """Get full document text by ID"""
    try:
        document = document_store.get_document(user_id, doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "success": True,
            "document": document
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{doc_id}/raw",
         tags=["Knowledge Base"])
async def get_document_raw(doc_id: str, user_id: str = Depends(get_current_user)):
    """Serve raw document as plain text for viewing in browser tab"""
    try:
        document = document_store.get_document(user_id, doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        return PlainTextResponse(
            content=document['content'],
            headers={"Content-Disposition": "inline"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving raw document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats",
         response_model=StatsResponse,
         tags=["System"])
async def get_stats(user_id: str = Depends(get_current_user)):
    """Get system statistics"""
    try:
        stats = orchestrator.get_stats()
        return StatsResponse(**stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dashboard-stats", tags=["System"])
async def get_dashboard_stats(user_id: str = Depends(get_current_user)):
    """Get combined dashboard statistics in a single call (Phase 5 optimization).

    Returns health, KB stats, study progress, and flashcard count in ONE response.
    Reduces 4 separate API calls to 1 for the dashboard.
    """
    try:
        # Combine all dashboard data into single response
        result = {
            "healthy": True,
            "kb_documents": 0,
            "flashcard_sets": 0,
            "total_quizzes": 0,
            "current_streak": 0,
            "topics_count": 0
        }

        # Get KB stats
        try:
            kb_stats = document_store.get_collection_stats(user_id)
            result["kb_documents"] = kb_stats.get("total_documents", 0)
        except Exception as e:
            logger.warning(f"Failed to get KB stats: {e}")

        # Get topics count
        try:
            topics = document_store.get_unique_topics(user_id)
            result["topics_count"] = len(topics)
        except Exception as e:
            logger.warning(f"Failed to get topics: {e}")

        return result

    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        # Return partial data rather than failing completely
        return {
            "healthy": False,
            "kb_documents": 0,
            "flashcard_sets": 0,
            "total_quizzes": 0,
            "current_streak": 0,
            "topics_count": 0,
            "error": str(e)
        }


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE
    )
