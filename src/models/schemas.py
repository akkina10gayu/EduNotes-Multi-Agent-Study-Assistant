"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class GenerateNotesRequest(BaseModel):
    """Request model for generating notes"""
    query: str = Field(..., description="Query text, URL, or topic")
    max_results: Optional[int] = Field(5, description="Maximum number of results to retrieve")
    summarization_mode: Optional[str] = Field("paragraph_summary", description="Format: 'paragraph_summary' for flowing paragraphs, 'important_points' for detailed bullets, 'key_highlights' for key terms")
    summary_length: Optional[str] = Field("auto", description="Length: 'auto' for smart sizing, 'brief' (~350 words), 'medium' (~800 words), 'detailed' (~2,300 words)")
    search_mode: Optional[str] = Field("auto", description="Search mode: 'auto' (KB first, web fallback), 'kb_only' (knowledge base only), 'web_search' (web search only), 'both' (KB and web search combined)")
    research_mode: Optional[bool] = Field(False, description="Research Mode: enables academic paper discovery, enhanced PDF extraction, and vision analysis of figures")
    
class GenerateNotesResponse(BaseModel):
    """Response model for generated notes"""
    success: bool
    query_type: str
    query: str
    notes: str
    sources_used: int
    from_kb: bool
    error: Optional[str] = None
    message: Optional[str] = None
    source_file: Optional[str] = None  # PDF filename if from PDF
    extracted_text: Optional[str] = None  # Phase 5: Extracted text for caching
    vision_data: Optional[str] = None  # JSON: figure images + descriptions for Research Mode

class UpdateKBRequest(BaseModel):
    """Request model for updating knowledge base"""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to add")
    
class UpdateKBResponse(BaseModel):
    """Response model for KB update"""
    success: bool
    documents_added: int
    error: Optional[str] = None

class SearchKBRequest(BaseModel):
    """Request model for searching knowledge base"""
    query: str = Field(..., description="Search query")
    k: Optional[int] = Field(5, description="Number of results")
    threshold: Optional[float] = Field(0.7, description="Similarity threshold")
    
class SearchKBResponse(BaseModel):
    """Response model for KB search"""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    count: int
    error: Optional[str] = None

class StatsResponse(BaseModel):
    """Response model for system statistics"""
    knowledge_base: Dict[str, Any]
    agents: Dict[str, str]
    
class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: datetime
    version: str