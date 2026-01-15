"""
FastAPI application for EduNotes
"""
from fastapi import FastAPI, HTTPException, Request
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
        logger.info(f"Generating notes for query: {body.query[:50]}...")
        
        # Process query through orchestrator
        result = await orchestrator.process(body.query)
        
        return GenerateNotesResponse(**result)
        
    except Exception as e:
        logger.error(f"Error generating notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/update-kb", 
          response_model=UpdateKBResponse, 
          tags=["Knowledge Base"])
@limiter.limit("10/minute")
async def update_knowledge_base(request: Request, body: UpdateKBRequest):
    """Update knowledge base with new documents"""
    try:
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
            
    except Exception as e:
        logger.error(f"Error updating KB: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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