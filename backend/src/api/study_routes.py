"""
API Routes for Study Features (Flashcards, Quizzes, Progress)
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.models.flashcard import (
    CreateFlashcardRequest, CreateFlashcardResponse,
    ReviewFlashcardRequest, ReviewFlashcardResponse,
    ListFlashcardSetsResponse
)
from src.models.quiz import (
    CreateQuizRequest, CreateQuizResponse,
    SubmitAnswerRequest, SubmitAnswerResponse,
    CompleteQuizRequest, CompleteQuizResponse,
    ListQuizzesResponse
)
from src.models.progress import (
    GetProgressResponse, RecordActivityRequest, RecordActivityResponse
)

from src.utils.flashcard_generator import get_flashcard_generator
from src.db.flashcard_store import FlashcardStore
from src.utils.quiz_generator import get_quiz_generator
from src.db import quiz_store
from src.db import progress_store as progress_db
from src.api.auth import get_current_user
from src.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/study", tags=["Study Features"])

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# =============================================================================
# FLASHCARD ENDPOINTS
# =============================================================================

@router.post("/flashcards/generate", response_model=CreateFlashcardResponse)
@limiter.limit("30/minute")
async def generate_flashcards(request: Request, body: CreateFlashcardRequest, user_id: str = Depends(get_current_user)):
    """Generate flashcards from content"""
    try:
        logger.info(f"Generating flashcards for topic: {body.topic}")

        generator = get_flashcard_generator()
        flashcard_set = generator.generate_flashcards(
            content=body.content,
            topic=body.topic,
            num_cards=body.num_cards,
            set_name=body.set_name
        )

        if flashcard_set:
            # Save to Supabase store
            store = FlashcardStore()
            store.save_set(user_id, topic=body.topic, cards=[c.model_dump() for c in flashcard_set.cards], source_content=body.content)

            # Record activity
            progress_db.record_activity(user_id, "flashcard_generated", body.topic, {})

            return CreateFlashcardResponse(
                success=True,
                flashcard_set=flashcard_set,
                message=f"Generated {len(flashcard_set.cards)} flashcards"
            )
        else:
            return CreateFlashcardResponse(
                success=False,
                message="Failed to generate flashcards",
                error="Generation failed"
            )

    except Exception as e:
        logger.error(f"Error generating flashcards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flashcards/sets", response_model=ListFlashcardSetsResponse)
async def list_flashcard_sets(user_id: str = Depends(get_current_user)):
    """List all flashcard sets"""
    try:
        store = FlashcardStore()
        sets = store.list_sets(user_id)

        return ListFlashcardSetsResponse(
            success=True,
            sets=sets,
            total_count=len(sets)
        )

    except Exception as e:
        logger.error(f"Error listing flashcard sets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flashcards/sets/{set_id}")
async def get_flashcard_set(set_id: str, user_id: str = Depends(get_current_user)):
    """Get a specific flashcard set"""
    try:
        store = FlashcardStore()
        flashcard_set = store.load_set(user_id, set_id)

        if flashcard_set:
            return {
                "success": True,
                "flashcard_set": flashcard_set
            }
        else:
            raise HTTPException(status_code=404, detail="Flashcard set not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flashcard set: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flashcards/review", response_model=ReviewFlashcardResponse)
async def review_flashcard(body: ReviewFlashcardRequest, user_id: str = Depends(get_current_user)):
    """Record a flashcard review"""
    try:
        store = FlashcardStore()
        card = store.update_card_review(user_id, body.set_id, body.card_id, body.correct)

        if card:
            # Record progress
            flashcard_set = store.load_set(user_id, body.set_id)
            if flashcard_set:
                topic = flashcard_set.get("topic", "unknown")
                progress_db.record_activity(user_id, "flashcard_review", topic, {"correct": body.correct})

            return ReviewFlashcardResponse(
                success=True,
                card=card,
                message="Review recorded"
            )
        else:
            return ReviewFlashcardResponse(
                success=False,
                card=None,
                message="Card not found"
            )

    except Exception as e:
        logger.error(f"Error recording review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/flashcards/sets/{set_id}")
async def delete_flashcard_set(set_id: str, user_id: str = Depends(get_current_user)):
    """Delete a flashcard set"""
    try:
        store = FlashcardStore()
        success = store.delete_set(user_id, set_id)

        return {
            "success": success,
            "message": "Flashcard set deleted" if success else "Failed to delete"
        }

    except Exception as e:
        logger.error(f"Error deleting flashcard set: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flashcards/export/all/anki")
async def export_all_flashcards_to_anki(user_id: str = Depends(get_current_user)):
    """Export all flashcard sets to Anki-compatible format"""
    try:
        from fastapi.responses import PlainTextResponse

        store = FlashcardStore()
        anki_export = store.export_all_to_anki(user_id)

        text = anki_export.get("text", "") if anki_export else ""

        return PlainTextResponse(
            content=text,
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=all_flashcards_anki.txt"}
        )

    except Exception as e:
        logger.error(f"Error exporting all flashcards to Anki: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flashcards/export/{set_id}/anki")
async def export_flashcards_to_anki(set_id: str, user_id: str = Depends(get_current_user)):
    """Export a flashcard set to Anki-compatible format"""
    try:
        from fastapi.responses import PlainTextResponse

        store = FlashcardStore()
        anki_export = store.export_to_anki(user_id, set_id)

        if anki_export:
            # Get set name for filename
            flashcard_set = store.load_set(user_id, set_id)
            filename = f"{flashcard_set.get('name', 'flashcards').replace(' ', '_')}_anki.txt" if flashcard_set else "flashcards_anki.txt"

            return PlainTextResponse(
                content=anki_export,
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            raise HTTPException(status_code=404, detail="Flashcard set not found or export failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting to Anki: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# QUIZ ENDPOINTS
# =============================================================================

@router.post("/quizzes/generate", response_model=CreateQuizResponse)
@limiter.limit("20/minute")
async def generate_quiz(request: Request, body: CreateQuizRequest, user_id: str = Depends(get_current_user)):
    """Generate a quiz from content"""
    try:
        logger.info(f"Generating quiz for topic: {body.topic}")

        generator = get_quiz_generator()
        quiz = generator.generate_quiz(
            content=body.content,
            topic=body.topic,
            num_questions=body.num_questions,
            title=body.title
        )

        if quiz:
            # Save to Supabase store
            quiz_store.save_quiz(user_id, topic=body.topic, questions=[q.model_dump() for q in quiz.questions])

            return CreateQuizResponse(
                success=True,
                quiz=quiz,
                message=f"Generated quiz with {len(quiz.questions)} questions"
            )
        else:
            return CreateQuizResponse(
                success=False,
                message="Failed to generate quiz",
                error="Generation failed"
            )

    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quizzes", response_model=ListQuizzesResponse)
async def list_quizzes(user_id: str = Depends(get_current_user)):
    """List all quizzes"""
    try:
        quizzes = quiz_store.list_quizzes(user_id)

        return ListQuizzesResponse(
            success=True,
            quizzes=quizzes,
            total_count=len(quizzes)
        )

    except Exception as e:
        logger.error(f"Error listing quizzes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quizzes/{quiz_id}")
async def get_quiz(quiz_id: str, user_id: str = Depends(get_current_user)):
    """Get a specific quiz"""
    try:
        quiz = quiz_store.load_quiz(user_id, quiz_id)

        if quiz:
            return {
                "success": True,
                "quiz": quiz
            }
        else:
            raise HTTPException(status_code=404, detail="Quiz not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quizzes/{quiz_id}/start")
async def start_quiz_attempt(quiz_id: str, user_id: str = Depends(get_current_user)):
    """Start a new quiz attempt"""
    try:
        attempt_id = quiz_store.start_attempt(user_id, quiz_id)
        quiz = quiz_store.load_quiz(user_id, quiz_id)

        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")

        questions = quiz.get("questions", [])

        return {
            "success": True,
            "attempt_id": attempt_id,
            "quiz_id": quiz_id,
            "total_questions": len(questions),
            "questions": [
                {
                    "id": q.get("id"),
                    "question": q.get("question"),
                    "options": q.get("options"),
                    "question_type": q.get("question_type")
                }
                for q in questions
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting quiz attempt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quizzes/submit-answer", response_model=SubmitAnswerResponse)
async def submit_quiz_answer(body: SubmitAnswerRequest, user_id: str = Depends(get_current_user)):
    """Submit an answer for a quiz question"""
    try:
        result = quiz_store.submit_answer(user_id, body.attempt_id, body.question_id, body.answer)

        return SubmitAnswerResponse(
            success=True,
            correct=result["is_correct"],
            correct_answer=result["correct_answer"],
            explanation=result.get("explanation"),
            message="Answer submitted"
        )

    except Exception as e:
        logger.error(f"Error submitting answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quizzes/complete", response_model=CompleteQuizResponse)
async def complete_quiz_attempt(body: CompleteQuizRequest, user_id: str = Depends(get_current_user)):
    """Complete a quiz attempt and get results"""
    try:
        result = quiz_store.complete_attempt(user_id, body.attempt_id)

        score = result["score"]
        total = result["total_questions"]

        # Fetch quiz topic from the attempt
        topic = "unknown"
        try:
            from src.db.supabase_client import get_supabase_client
            sb = get_supabase_client()
            attempt_row = sb.table("quiz_attempts").select("quiz_id").eq("id", body.attempt_id).execute()
            if attempt_row.data:
                quiz_id = attempt_row.data[0]["quiz_id"]
                quiz_row = sb.table("quizzes").select("topic").eq("id", quiz_id).execute()
                if quiz_row.data:
                    topic = quiz_row.data[0]["topic"]
        except Exception:
            pass

        # Record progress
        progress_db.record_activity(user_id, "quiz_completed", topic, {"score": score, "total": total})

        return CompleteQuizResponse(
            success=True,
            score=score,
            correct_count=result["correct_count"],
            total_questions=total,
            results={},
            detailed_results=result.get("detailed_results", []),
            message="Quiz completed"
        )

    except Exception as e:
        logger.error(f"Error completing quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/quizzes/{quiz_id}")
async def delete_quiz(quiz_id: str, user_id: str = Depends(get_current_user)):
    """Delete a quiz"""
    try:
        success = quiz_store.delete_quiz(user_id, quiz_id)

        return {
            "success": success,
            "message": "Quiz deleted" if success else "Failed to delete"
        }

    except Exception as e:
        logger.error(f"Error deleting quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PROGRESS ENDPOINTS
# =============================================================================

@router.get("/progress")
async def get_progress(user_id: str = Depends(get_current_user)):
    """Get user study progress"""
    try:
        statistics = progress_db.get_statistics(user_id)
        topic_rankings = progress_db.get_topic_rankings(user_id)
        weekly_summary = progress_db.get_weekly_summary(user_id)
        recent_activities = progress_db.get_recent_activities(user_id, limit=10)

        # Fetch streak data
        from src.db.supabase_client import get_supabase_client
        sb = get_supabase_client()
        streak_result = sb.table("study_streaks").select("*").eq("user_id", user_id).execute()
        streak_row = streak_result.data[0] if streak_result.data else None
        streak = {
            "current_streak": streak_row["current_streak"] if streak_row else 0,
            "longest_streak": streak_row["best_streak"] if streak_row else 0,
            "last_study_date": streak_row.get("last_activity_date") if streak_row else None,
        }

        return {
            "success": True,
            "statistics": statistics,
            "topic_rankings": topic_rankings,
            "weekly_summary": weekly_summary,
            "recent_activities": recent_activities,
            "streak": streak
        }

    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/progress/record")
async def record_activity(body: RecordActivityRequest, user_id: str = Depends(get_current_user)):
    """Record a study activity"""
    try:
        progress_db.record_activity(user_id, body.activity_type, body.topic, body.details)

        return {
            "success": True,
            "message": "Activity recorded"
        }

    except Exception as e:
        logger.error(f"Error recording activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/stats")
async def get_study_stats(user_id: str = Depends(get_current_user)):
    """Get detailed study statistics"""
    try:
        return {
            "success": True,
            "statistics": progress_db.get_statistics(user_id),
            "topic_rankings": progress_db.get_topic_rankings(user_id),
            "weekly_summary": progress_db.get_weekly_summary(user_id)
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/progress/reset")
async def reset_progress(user_id: str = Depends(get_current_user)):
    """Reset all progress (use with caution!)"""
    try:
        success = progress_db.reset_progress(user_id)

        return {
            "success": success,
            "message": "Progress reset" if success else "Failed to reset"
        }

    except Exception as e:
        logger.error(f"Error resetting progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))
