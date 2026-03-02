"""
FastAPI routes for the Conversational AI Chat feature.
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException

from src.models.chat import (
    ChatRequest, ChatResponse,
    ResearchStartRequest, ResearchContinueRequest,
    ResearchSectionRequest, ResearchAssembleRequest,
    ResearchResponse,
)
from src.agents.chat_agent import ChatAgent
from src.agents.research_writer import ResearchWriter
from src.utils.conversation_store import get_conversation_store
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Lazy-initialized singletons
_chat_agent = None
_research_writer = None
_research_sessions: dict = {}


def _get_chat_agent() -> ChatAgent:
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = ChatAgent()
    return _chat_agent


def _get_research_writer() -> ResearchWriter:
    global _research_writer
    if _research_writer is None:
        _research_writer = ResearchWriter()
    return _research_writer


# ======================================================================
# Chat endpoints
# ======================================================================

@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """Send a chat message and get an AI response."""
    try:
        agent = _get_chat_agent()

        result = await agent.process({
            "message": request.message,
            "mode": request.mode,
            "history": request.history,
            "use_kb": request.use_kb,
            "explain_level": request.explain_level,
            "explain_style": request.explain_style,
            "analogy_domain": request.analogy_domain,
            "answer_depth": request.answer_depth,
            "compare_concept_2": request.compare_concept_2,
        })

        if not result.get("success"):
            return ChatResponse(
                success=False,
                response="",
                error=result.get("error", "Unknown error"),
                mode=request.mode,
            )

        # Persist to session
        session_id = request.session_id or str(uuid.uuid4())[:8]
        store = get_conversation_store()
        session = store.load_session(session_id)
        if not session:
            title = request.message[:50]
            if len(request.message) > 50:
                title += "..."
            session = {
                "id": session_id,
                "title": title,
                "mode": request.mode,
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

        now = datetime.now().isoformat()
        session["messages"].append({
            "role": "user",
            "content": request.message,
            "timestamp": now,
        })
        session["messages"].append({
            "role": "assistant",
            "content": result["response"],
            "timestamp": now,
            "metadata": {
                "provider": result.get("provider_used"),
                "sources": result.get("sources", []),
                "suggestions": result.get("suggestions", []),
            },
        })
        store.save_session(session)

        return ChatResponse(
            success=True,
            response=result["response"],
            suggestions=result.get("suggestions", []),
            sources=result.get("sources", []),
            session_id=session_id,
            mode=request.mode,
            provider_used=result.get("provider_used", "unknown"),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            success=False, response="", error=str(e), mode=request.mode
        )


# ======================================================================
# Research paper endpoints
# ======================================================================

@router.post("/research/start", response_model=ResearchResponse)
async def research_start(request: ResearchStartRequest):
    """Start a new research paper session."""
    try:
        writer = _get_research_writer()
        session_id = f"research_{uuid.uuid4().hex[:6]}"

        result = await writer.process({
            "action": "analyze",
            "topic": request.topic,
            "context": request.initial_context or "",
            "paper_type": request.paper_type,
        })

        # Store session state in memory
        _research_sessions[session_id] = {
            "topic": request.topic,
            "paper_type": request.paper_type,
            "initial_context": request.initial_context or "",
            "gathered_info": {},
            "outline": result.get("outline", []),
            "sections": {},
            "stage": result.get("stage", "gathering"),
        }

        return ResearchResponse(
            success=True,
            session_id=session_id,
            stage=result.get("stage", "gathering"),
            analysis=result.get("analysis"),
            questions=result.get("questions"),
            sufficiency_score=result.get("sufficiency_score", 0.0),
            info_available=result.get("info_available", []),
            info_missing=result.get("info_missing", []),
            related_papers=result.get("related_papers"),
        )
    except Exception as e:
        logger.error(f"Research start error: {e}")
        return ResearchResponse(
            success=False, session_id="", stage="gathering", error=str(e)
        )


@router.post("/research/continue", response_model=ResearchResponse)
async def research_continue(request: ResearchContinueRequest):
    """Provide answers to follow-up questions and re-analyze sufficiency."""
    session = _research_sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    try:
        session["gathered_info"].update(request.answers)

        writer = _get_research_writer()
        result = await writer.process({
            "action": "continue",
            "topic": session["topic"],
            "paper_type": session["paper_type"],
            "initial_context": session["initial_context"],
            "gathered_info": session["gathered_info"],
        })

        session["stage"] = result.get("stage", "gathering")
        if result.get("outline"):
            session["outline"] = result["outline"]

        return ResearchResponse(
            success=True,
            session_id=request.session_id,
            stage=result.get("stage", "gathering"),
            analysis=result.get("analysis"),
            questions=result.get("questions"),
            outline=result.get("outline"),
            sufficiency_score=result.get("sufficiency_score", 0.0),
            info_available=result.get("info_available", []),
            info_missing=result.get("info_missing", []),
            related_papers=result.get("related_papers"),
        )
    except Exception as e:
        logger.error(f"Research continue error: {e}")
        return ResearchResponse(
            success=False, session_id=request.session_id,
            stage="gathering", error=str(e),
        )


@router.post("/research/generate-section", response_model=ResearchResponse)
async def research_generate_section(request: ResearchSectionRequest):
    """Generate a specific paper section."""
    session = _research_sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    try:
        # Build full context
        context_parts = []
        if session["initial_context"]:
            context_parts.append(session["initial_context"])
        for key, value in session["gathered_info"].items():
            context_parts.append(f"{key}: {value}")

        writer = _get_research_writer()
        result = await writer.process({
            "action": "section",
            "topic": session["topic"],
            "paper_type": session["paper_type"],
            "section_name": request.section_name,
            "context": "\n\n".join(context_parts),
            "outline": session["outline"],
            "previous_sections": session["sections"],
            "additional_instructions": request.additional_instructions,
        })

        if result.get("success") and result.get("content"):
            session["sections"][request.section_name] = result["content"]

        return ResearchResponse(
            success=True,
            session_id=request.session_id,
            stage="writing",
            content=result.get("content"),
        )
    except Exception as e:
        logger.error(f"Research section error: {e}")
        return ResearchResponse(
            success=False, session_id=request.session_id,
            stage="writing", error=str(e),
        )


@router.post("/research/assemble", response_model=ResearchResponse)
async def research_assemble(request: ResearchAssembleRequest):
    """Assemble all generated sections into the final paper."""
    session = _research_sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    try:
        writer = _get_research_writer()
        result = await writer.process({
            "action": "assemble",
            "topic": session["topic"],
            "sections": session["sections"],
            "generate_abstract": request.generate_abstract,
        })

        return ResearchResponse(
            success=True,
            session_id=request.session_id,
            stage="complete",
            full_paper=result.get("full_paper"),
            word_count=result.get("word_count"),
            is_complete=True,
        )
    except Exception as e:
        logger.error(f"Research assemble error: {e}")
        return ResearchResponse(
            success=False, session_id=request.session_id,
            stage="polishing", error=str(e),
        )


# ======================================================================
# Session management endpoints
# ======================================================================

@router.get("/sessions")
async def list_sessions():
    """List all saved chat sessions."""
    store = get_conversation_store()
    return {"sessions": store.list_sessions()}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full chat session with all messages."""
    store = get_conversation_store()
    session = store.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    store = get_conversation_store()
    if store.delete_session(session_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/sessions/{session_id}/export")
async def export_session(session_id: str):
    """Export a session as markdown."""
    store = get_conversation_store()
    md = store.export_session_markdown(session_id)
    if md is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session = store.load_session(session_id)
    title = session.get("title", "chat_export") if session else "chat_export"
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    return {"markdown": md, "filename": f"{safe_title[:30]}.md"}
