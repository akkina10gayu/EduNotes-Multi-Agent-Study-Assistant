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