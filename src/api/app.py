"""
FastAPI application for EduNotes
"""
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from datetime import datetime
import asyncio

from src.models.schemas import (
    GenerateNotesRequest, GenerateNotesResponse,
    UpdateKBRequest, UpdateKBResponse,
    SearchKBRequest, SearchKBResponse,
    StatsResponse, HealthResponse
)
from src.agents.orchestrator import Orchestrator
from src.knowledge_base.vector_store import VectorStore
from src.knowledge_base.document_processor import DocumentProcessor
from src.utils.logger import get_logger
from src.utils.pdf_processor import get_pdf_processor
from src.api.study_routes import router as study_router
from config import settings

logger = get_logger(__name__)

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
vector_store = VectorStore()
doc_processor = DocumentProcessor()

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
async def generate_notes(request: Request, body: GenerateNotesRequest):
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

        logger.info(f"Generating notes for query: {query[:50]}... Mode: {body.summarization_mode}")

        # Process query through orchestrator
        result = await orchestrator.process(query, summarization_mode=body.summarization_mode)

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
async def update_knowledge_base(request: Request, body: UpdateKBRequest):
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

        # Process documents
        processed = doc_processor.process_batch(body.documents)

        # Add to vector store
        if processed:
            success = vector_store.add_documents(processed)

            return UpdateKBResponse(
                success=success,
                documents_added=len(processed)
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
async def search_knowledge_base(request: Request, body: SearchKBRequest):
    """Search the knowledge base"""
    try:
        logger.info(f"Searching KB for: {body.query[:50]}...")
        
        results = vector_store.search(
            query=body.query,
            k=body.k,
            score_threshold=body.threshold
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
    file: UploadFile = File(...),
    summarization_mode: str = "detailed"
):
    """Process PDF file and generate study notes"""
    try:
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

        logger.info(f"Processing PDF: {file.filename} ({file_size_mb:.2f}MB)")

        # Extract text from PDF
        pdf_processor = get_pdf_processor()
        extracted_text = pdf_processor.extract_text_from_bytes(content)

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

        logger.info(f"Extracted {len(extracted_text)} characters from PDF")
        logger.info(f"Processing PDF with summarization_mode='{summarization_mode}'")

        # Process through orchestrator as text query
        result = await orchestrator.process(extracted_text, summarization_mode=summarization_mode)

        logger.info(f"PDF processing completed - Success: {result.get('success', False)}")

        # Add PDF filename to result
        if result.get('success'):
            result['source_file'] = file.filename

        return GenerateNotesResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/v1/topics",
         tags=["Knowledge Base"])
async def get_topics():
    """Get available topics from knowledge base"""
    try:
        # Get unique topics from vector store
        topics = vector_store.get_unique_topics()

        return {
            "success": True,
            "topics": topics[:20],  # Limit to top 20 topics
            "total_topics": len(topics)
        }

    except Exception as e:
        logger.error(f"Error getting topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats",
         response_model=StatsResponse,
         tags=["System"])
async def get_stats():
    """Get system statistics"""
    try:
        stats = orchestrator.get_stats()
        return StatsResponse(**stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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