"""Quiz store backed by Supabase.

Consolidates all quiz persistence (quizzes, questions, attempts, answers)
into a single module that talks to the Supabase tables defined in the
initial migration.
"""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _supabase():
    """Return the shared Supabase client (lazy import to avoid circular deps)."""
    from src.db.supabase_client import get_supabase_client
    return get_supabase_client()


# ---------------------------------------------------------------------------
# Quiz CRUD
# ---------------------------------------------------------------------------


def save_quiz(
    user_id: str,
    topic: str,
    questions: list[dict[str, Any]],
    source_content: str = "",
) -> dict[str, Any]:
    """Create a quiz with its questions.

    *questions* is a list of dicts, each containing:
        question, question_type, options (list), correct_answer, explanation

    Returns a dict with id, title, topic, questions, created_at.
    """
    sb = _supabase()

    # 1. Insert the quiz row
    quiz_payload = {
        "user_id": user_id,
        "topic": topic,
        "source_content": source_content,
        "question_count": 0,
    }
    result = sb.table("quizzes").insert(quiz_payload).execute()
    quiz_row = result.data[0]
    quiz_id = quiz_row["id"]
    logger.info("Created quiz %s (topic=%s) for user %s", quiz_id, topic, user_id)

    # 2. Bulk-insert questions
    question_rows = []
    for idx, q in enumerate(questions):
        question_rows.append({
            "quiz_id": quiz_id,
            "question": q["question"],
            "question_type": q.get("question_type", "multiple_choice"),
            "options": q.get("options"),
            "correct_answer": q["correct_answer"],
            "explanation": q.get("explanation", ""),
            "order_index": idx,
        })

    inserted_questions: list[dict[str, Any]] = []
    if question_rows:
        q_result = sb.table("quiz_questions").insert(question_rows).execute()
        inserted_questions = q_result.data
        logger.info("Inserted %d questions for quiz %s", len(inserted_questions), quiz_id)

    # 3. Update question_count on the quiz
    sb.table("quizzes").update(
        {"question_count": len(inserted_questions)}
    ).eq("id", quiz_id).execute()

    return {
        "id": quiz_id,
        "title": topic,
        "topic": topic,
        "questions": inserted_questions,
        "created_at": quiz_row["created_at"],
    }


def load_quiz(user_id: str, quiz_id: str) -> dict[str, Any] | None:
    """Load a quiz with all its questions (ordered by order_index).

    Returns None if the quiz does not exist or does not belong to *user_id*.
    """
    sb = _supabase()

    quiz_result = (
        sb.table("quizzes")
        .select("*")
        .eq("id", quiz_id)
        .eq("user_id", user_id)
        .execute()
    )

    if not quiz_result.data:
        logger.warning("Quiz %s not found for user %s", quiz_id, user_id)
        return None

    quiz_row = quiz_result.data[0]

    questions_result = (
        sb.table("quiz_questions")
        .select("*")
        .eq("quiz_id", quiz_id)
        .order("order_index")
        .execute()
    )

    return {
        "id": quiz_row["id"],
        "title": quiz_row["topic"],
        "topic": quiz_row["topic"],
        "questions": questions_result.data,
        "question_count": quiz_row["question_count"],
        "created_at": quiz_row["created_at"],
    }


def list_quizzes(user_id: str) -> list[dict[str, Any]]:
    """List all quizzes for *user_id* (most recent first)."""
    sb = _supabase()

    result = (
        sb.table("quizzes")
        .select("id, topic, question_count, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    quizzes = [
        {
            "id": row["id"],
            "title": row["topic"],
            "question_count": row["question_count"],
            "created_at": row["created_at"],
        }
        for row in result.data
    ]
    logger.info("Listed %d quizzes for user %s", len(quizzes), user_id)
    return quizzes


def delete_quiz(user_id: str, quiz_id: str) -> bool:
    """Delete a quiz and all related data (cascade)."""
    sb = _supabase()

    logger.info("Deleting quiz %s for user %s", quiz_id, user_id)
    sb.table("quizzes").delete().eq("id", quiz_id).eq("user_id", user_id).execute()
    logger.info("Quiz %s deleted", quiz_id)
    return True


# ---------------------------------------------------------------------------
# Attempt lifecycle
# ---------------------------------------------------------------------------


def start_attempt(user_id: str, quiz_id: str) -> str:
    """Start a new quiz attempt.

    Verifies the quiz belongs to *user_id*, then creates an attempt row.
    Returns the attempt id.

    Raises ``ValueError`` if the quiz does not belong to the user.
    """
    sb = _supabase()

    # Verify ownership
    quiz_result = (
        sb.table("quizzes")
        .select("id, question_count")
        .eq("id", quiz_id)
        .eq("user_id", user_id)
        .execute()
    )

    if not quiz_result.data:
        raise ValueError(f"Quiz {quiz_id} not found for user {user_id}")

    total_questions = quiz_result.data[0]["question_count"]

    attempt_payload = {
        "quiz_id": quiz_id,
        "user_id": user_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": total_questions,
    }
    result = sb.table("quiz_attempts").insert(attempt_payload).execute()
    attempt_id = result.data[0]["id"]
    logger.info(
        "Started attempt %s for quiz %s (user %s, %d questions)",
        attempt_id,
        quiz_id,
        user_id,
        total_questions,
    )
    return attempt_id


def submit_answer(
    user_id: str,
    attempt_id: str,
    question_id: str,
    answer: str,
) -> dict[str, Any]:
    """Submit an answer for a single question within an attempt.

    Returns a dict with is_correct, correct_answer, explanation.
    """
    sb = _supabase()

    # Look up the correct answer and explanation
    q_result = (
        sb.table("quiz_questions")
        .select("correct_answer, explanation")
        .eq("id", question_id)
        .execute()
    )

    if not q_result.data:
        raise ValueError(f"Question {question_id} not found")

    correct_answer = q_result.data[0]["correct_answer"]
    explanation = q_result.data[0].get("explanation", "")
    is_correct = answer.strip().lower() == correct_answer.strip().lower()

    # Insert the answer row
    answer_payload = {
        "attempt_id": attempt_id,
        "question_id": question_id,
        "user_answer": answer,
        "is_correct": is_correct,
    }
    sb.table("quiz_answers").insert(answer_payload).execute()
    logger.info(
        "Answer submitted for question %s in attempt %s — correct: %s",
        question_id,
        attempt_id,
        is_correct,
    )

    return {
        "is_correct": is_correct,
        "correct_answer": correct_answer,
        "explanation": explanation,
    }


def complete_attempt(user_id: str, attempt_id: str) -> dict[str, Any]:
    """Finalise an attempt: compute score and return detailed results.

    Returns a dict with score (percentage), correct_count, total_questions,
    and detailed_results (per question).
    """
    sb = _supabase()

    # Fetch all answers for this attempt
    answers_result = (
        sb.table("quiz_answers")
        .select("question_id, user_answer, is_correct")
        .eq("attempt_id", attempt_id)
        .execute()
    )

    answers = answers_result.data
    total_questions = len(answers)
    correct_count = sum(1 for a in answers if a["is_correct"])
    score = round((correct_count / total_questions * 100), 2) if total_questions else 0.0

    # Update the attempt row
    sb.table("quiz_attempts").update({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "score": score,
        "correct_count": correct_count,
        "total_questions": total_questions,
    }).eq("id", attempt_id).execute()

    logger.info(
        "Completed attempt %s — score %.1f%% (%d/%d)",
        attempt_id,
        score,
        correct_count,
        total_questions,
    )

    # Build detailed results by joining with quiz_questions
    detailed_results = _build_detailed_results(sb, answers)

    return {
        "score": score,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "detailed_results": detailed_results,
    }


def get_attempt_results(user_id: str, attempt_id: str) -> list[dict[str, Any]]:
    """Return detailed per-question results for a completed attempt.

    Each entry contains: question_text, options, user_answer, correct_answer,
    is_correct, explanation.
    """
    sb = _supabase()

    # Verify the attempt belongs to the user
    attempt_result = (
        sb.table("quiz_attempts")
        .select("id")
        .eq("id", attempt_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not attempt_result.data:
        logger.warning("Attempt %s not found for user %s", attempt_id, user_id)
        return []

    answers_result = (
        sb.table("quiz_answers")
        .select("question_id, user_answer, is_correct")
        .eq("attempt_id", attempt_id)
        .execute()
    )

    return _build_detailed_results(sb, answers_result.data)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_detailed_results(
    sb: Any,
    answers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich answer rows with question text, options, and explanation."""
    if not answers:
        return []

    question_ids = [a["question_id"] for a in answers]

    # Fetch all relevant questions in one go
    q_result = (
        sb.table("quiz_questions")
        .select("id, question, options, correct_answer, explanation, order_index")
        .in_("id", question_ids)
        .execute()
    )

    q_map = {q["id"]: q for q in q_result.data}

    detailed: list[dict[str, Any]] = []
    for a in answers:
        q = q_map.get(a["question_id"], {})
        detailed.append({
            "question_text": q.get("question", ""),
            "options": q.get("options"),
            "user_answer": a["user_answer"],
            "correct_answer": q.get("correct_answer", ""),
            "is_correct": a["is_correct"],
            "explanation": q.get("explanation", ""),
        })

    # Sort by original question order
    detailed.sort(key=lambda d: q_map.get(
        next((a["question_id"] for a in answers
              if a["user_answer"] == d["user_answer"]
              and a["is_correct"] == d["is_correct"]), ""),
        {},
    ).get("order_index", 0))

    return detailed
