"""
Pydantic models for the Conversational AI Chat feature.
"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


class ChatMode(str, Enum):
    CHAT = "chat"
    ANSWER_WRITER = "answer_writer"
    EXPLAIN = "explain"
    COMPARE = "compare"
    SOCRATIC = "socratic"
    PAPER_ANALYSIS = "paper_analysis"


class ExplainLevel(str, Enum):
    ELI5 = "eli5"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ExplainStyle(str, Enum):
    TECHNICAL = "technical"
    ANALOGY = "analogy"
    VISUAL = "visual"


class AnswerDepth(str, Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


class PaperStage(str, Enum):
    GATHERING = "gathering"
    OUTLINING = "outlining"
    WRITING = "writing"
    POLISHING = "polishing"
    COMPLETE = "complete"


class PaperType(str, Enum):
    RESEARCH = "research"
    SURVEY = "survey"
    CASE_STUDY = "case_study"


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = None


class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = "New Chat"
    mode: str = "chat"
    messages: List[ChatMessage] = []
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=50000)
    session_id: Optional[str] = None
    mode: ChatMode = ChatMode.CHAT
    use_kb: bool = True
    explain_level: Optional[ExplainLevel] = None
    explain_style: Optional[ExplainStyle] = None
    analogy_domain: Optional[str] = None
    answer_depth: Optional[AnswerDepth] = None
    compare_concept_2: Optional[str] = None
    history: List[Dict[str, str]] = []


class ChatResponse(BaseModel):
    success: bool
    response: str
    suggestions: List[str] = []
    sources: List[Dict[str, Any]] = []
    session_id: Optional[str] = None
    mode: str = "chat"
    provider_used: str = "unknown"
    tokens_used: Optional[int] = None
    error: Optional[str] = None


class ResearchStartRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    initial_context: Optional[str] = Field(None, max_length=50000)
    paper_type: PaperType = PaperType.RESEARCH


class ResearchContinueRequest(BaseModel):
    session_id: str
    answers: Dict[str, str] = {}


class ResearchSectionRequest(BaseModel):
    session_id: str
    section_name: str
    additional_instructions: Optional[str] = None


class ResearchAssembleRequest(BaseModel):
    session_id: str
    generate_abstract: bool = True


class ResearchResponse(BaseModel):
    success: bool
    session_id: str
    stage: str
    content: Optional[str] = None
    analysis: Optional[str] = None
    questions: Optional[List[str]] = None
    outline: Optional[List[str]] = None
    sufficiency_score: float = 0.0
    info_available: List[str] = []
    info_missing: List[str] = []
    related_papers: Optional[List[Dict]] = None
    full_paper: Optional[str] = None
    word_count: Optional[int] = None
    is_complete: bool = False
    error: Optional[str] = None
