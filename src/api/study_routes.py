"""
API Routes for Study Features (Flashcards, Quizzes, Progress)
"""
from fastapi import APIRouter, HTTPException, Request
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
from src.utils.flashcard_store import get_flashcard_store
from src.utils.quiz_generator import get_quiz_generator, get_quiz_store
from src.utils.progress_store import get_progress_store
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
async def generate_flashcards(request: Request, body: CreateFlashcardRequest):
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
            # Record activity
            progress_store = get_progress_store()
            progress_store.record_note_generated(body.topic)

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
async def list_flashcard_sets():
    """List all flashcard sets"""
    try:
        store = get_flashcard_store()
        sets = store.list_sets()

        return ListFlashcardSetsResponse(
            success=True,
            sets=sets,
            total_count=len(sets)
        )

    except Exception as e:
        logger.error(f"Error listing flashcard sets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flashcards/sets/{set_id}")
async def get_flashcard_set(set_id: str):
    """Get a specific flashcard set"""
    try:
        store = get_flashcard_store()
        flashcard_set = store.load_set(set_id)

        if flashcard_set:
            return {
                "success": True,
                "flashcard_set": flashcard_set.model_dump()
            }
        else:
            raise HTTPException(status_code=404, detail="Flashcard set not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flashcard set: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flashcards/review", response_model=ReviewFlashcardResponse)
async def review_flashcard(body: ReviewFlashcardRequest):
    """Record a flashcard review"""
    try:
        store = get_flashcard_store()
        card = store.update_card_review(body.set_id, body.card_id, body.correct)

        if card:
            # Record progress
            flashcard_set = store.load_set(body.set_id)
            if flashcard_set:
                progress_store = get_progress_store()
                progress_store.record_flashcard_review(flashcard_set.topic, body.correct)

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
async def delete_flashcard_set(set_id: str):
    """Delete a flashcard set"""
    try:
        store = get_flashcard_store()
        success = store.delete_set(set_id)

        return {
            "success": success,
            "message": "Flashcard set deleted" if success else "Failed to delete"
        }

    except Exception as e:
        logger.error(f"Error deleting flashcard set: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flashcards/export/{set_id}/anki")
async def export_flashcards_to_anki(set_id: str):
    """Export a flashcard set to Anki-compatible format"""
    try:
        from fastapi.responses import PlainTextResponse

        store = get_flashcard_store()
        anki_export = store.export_to_anki(set_id)

        if anki_export:
            # Get set name for filename
            flashcard_set = store.load_set(set_id)
            filename = f"{flashcard_set.name.replace(' ', '_')}_anki.txt" if flashcard_set else "flashcards_anki.txt"

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


@router.get("/flashcards/export/all/anki")
async def export_all_flashcards_to_anki():
    """Export all flashcard sets to Anki-compatible format"""
    try:
        from fastapi.responses import PlainTextResponse

        store = get_flashcard_store()
        anki_export = store.export_all_to_anki()

        if anki_export:
            return PlainTextResponse(
                content=anki_export,
                media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=all_flashcards_anki.txt"}
            )
        else:
            return PlainTextResponse(
                content="",
                media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=all_flashcards_anki.txt"}
            )

    except Exception as e:
        logger.error(f"Error exporting all flashcards to Anki: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# QUIZ ENDPOINTS
# =============================================================================

@router.post("/quizzes/generate", response_model=CreateQuizResponse)
@limiter.limit("20/minute")
async def generate_quiz(request: Request, body: CreateQuizRequest):
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
async def list_quizzes():
    """List all quizzes"""
    try:
        store = get_quiz_store()
        quizzes = store.list_quizzes()

        return ListQuizzesResponse(
            success=True,
            quizzes=quizzes,
            total_count=len(quizzes)
        )

    except Exception as e:
        logger.error(f"Error listing quizzes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quizzes/{quiz_id}")
async def get_quiz(quiz_id: str):
    """Get a specific quiz"""
    try:
        store = get_quiz_store()
        quiz = store.load_quiz(quiz_id)

        if quiz:
            return {
                "success": True,
                "quiz": quiz.model_dump()
            }
        else:
            raise HTTPException(status_code=404, detail="Quiz not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quizzes/{quiz_id}/start")
async def start_quiz_attempt(quiz_id: str):
    """Start a new quiz attempt"""
    try:
        store = get_quiz_store()
        quiz = store.load_quiz(quiz_id)

        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")

        attempt = quiz.start_attempt()
        store.save_quiz(quiz)

        return {
            "success": True,
            "attempt_id": attempt.id,
            "quiz_id": quiz_id,
            "total_questions": len(quiz.questions),
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "options": q.options,
                    "question_type": q.question_type.value
                }
                for q in quiz.questions
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting quiz attempt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quizzes/submit-answer", response_model=SubmitAnswerResponse)
async def submit_quiz_answer(body: SubmitAnswerRequest):
    """Submit an answer for a quiz question"""
    try:
        generator = get_quiz_generator()
        result = generator.submit_answer(
            body.quiz_id,
            body.attempt_id,
            body.question_id,
            body.answer
        )

        if result.get("success"):
            return SubmitAnswerResponse(
                success=True,
                correct=result["correct"],
                correct_answer=result["correct_answer"],
                explanation=result.get("explanation"),
                message="Answer submitted"
            )
        else:
            return SubmitAnswerResponse(
                success=False,
                correct=False,
                correct_answer="",
                message=result.get("error", "Failed to submit answer")
            )

    except Exception as e:
        logger.error(f"Error submitting answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quizzes/complete", response_model=CompleteQuizResponse)
async def complete_quiz_attempt(body: CompleteQuizRequest):
    """Complete a quiz attempt and get results"""
    try:
        generator = get_quiz_generator()
        result = generator.complete_attempt(body.quiz_id, body.attempt_id)

        if result.get("success"):
            # Record progress
            store = get_quiz_store()
            quiz = store.load_quiz(body.quiz_id)
            if quiz:
                progress_store = get_progress_store()
                progress_store.record_quiz_completed(
                    quiz.topic,
                    result["score"],
                    result["total_questions"],
                    result["correct_count"]
                )

            return CompleteQuizResponse(
                success=True,
                score=result["score"],
                correct_count=result["correct_count"],
                total_questions=result["total_questions"],
                results=result["results"],
                message="Quiz completed"
            )
        else:
            return CompleteQuizResponse(
                success=False,
                score=0,
                correct_count=0,
                total_questions=0,
                results={},
                message=result.get("error", "Failed to complete quiz")
            )

    except Exception as e:
        logger.error(f"Error completing quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/quizzes/{quiz_id}")
async def delete_quiz(quiz_id: str):
    """Delete a quiz"""
    try:
        store = get_quiz_store()
        success = store.delete_quiz(quiz_id)

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

@router.get("/progress", response_model=GetProgressResponse)
async def get_progress():
    """Get user study progress"""
    try:
        store = get_progress_store()
        progress = store.load_progress()

        return GetProgressResponse(
            success=True,
            overall_stats=progress.get_overall_stats(),
            topic_rankings=progress.get_topic_rankings(),
            weekly_summary=progress.get_weekly_summary(),
            recent_activities=store.get_recent_activities(limit=10),
            streak={
                "current_streak": progress.streak.current_streak,
                "longest_streak": progress.streak.longest_streak,
                "last_study_date": str(progress.streak.last_study_date) if progress.streak.last_study_date else None
            }
        )

    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/progress/record", response_model=RecordActivityResponse)
async def record_activity(body: RecordActivityRequest):
    """Record a study activity"""
    try:
        store = get_progress_store()
        progress = store.record_activity(
            body.activity_type,
            body.topic,
            body.details
        )

        return RecordActivityResponse(
            success=True,
            message="Activity recorded",
            current_streak=progress.streak.current_streak
        )

    except Exception as e:
        logger.error(f"Error recording activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/stats")
async def get_study_stats():
    """Get detailed study statistics"""
    try:
        store = get_progress_store()

        return {
            "success": True,
            "statistics": store.get_statistics(),
            "topic_rankings": store.get_topic_rankings(),
            "weekly_summary": store.get_weekly_summary()
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/progress/reset")
async def reset_progress():
    """Reset all progress (use with caution!)"""
    try:
        store = get_progress_store()
        success = store.reset_progress()

        return {
            "success": success,
            "message": "Progress reset" if success else "Failed to reset"
        }

    except Exception as e:
        logger.error(f"Error resetting progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))
