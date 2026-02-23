"""Progress store backed by Supabase.

Replaces the legacy JSON-file-based ProgressStore with an append-only
activity log (study_activities) and a per-user streak row (study_streaks).
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _today() -> date:
    return datetime.now(timezone.utc).date()


def _monday_of_current_week() -> datetime:
    """Return midnight UTC of the Monday in the current ISO week."""
    today = _today()
    monday = today - timedelta(days=today.weekday())
    return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)


def _mastery_level(activity_count: int) -> str:
    if activity_count < 5:
        return "Beginner"
    if activity_count < 15:
        return "Intermediate"
    if activity_count < 30:
        return "Advanced"
    return "Expert"


def _activity_summary(activity_type: str, topic: str, metadata: dict | None) -> str:
    """Build a human-readable summary string for an activity."""
    meta = metadata or {}
    if activity_type == "note_generated":
        return f"Generated notes on {topic}" if topic else "Generated notes"
    if activity_type == "flashcard_review":
        result = "correctly" if meta.get("correct") else "incorrectly"
        return f"Reviewed flashcard on {topic} ({result})"
    if activity_type == "quiz_completed":
        score = meta.get("score", "?")
        total = meta.get("total", "?")
        return f"Completed quiz on {topic} — {score}/{total}"
    if activity_type == "kb_search":
        return f"Searched knowledge base for {topic}" if topic else "Searched knowledge base"
    return f"{activity_type} on {topic}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_activity(
    user_id: str,
    activity_type: str,
    topic: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append an activity row and upsert the user's streak."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    # 1. Insert activity
    payload = {
        "user_id": user_id,
        "activity_type": activity_type,
        "topic": topic,
        "metadata": metadata or {},
    }
    supabase.table("study_activities").insert(payload).execute()
    logger.info("Recorded activity %s for user %s (topic=%s)", activity_type, user_id, topic)

    # 2. Upsert streak
    today = _today()
    yesterday = today - timedelta(days=1)

    # Fetch current streak row (may not exist yet)
    streak_result = (
        supabase.table("study_streaks")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )
    streak_row = streak_result.data[0] if streak_result.data else None

    if streak_row is None:
        # First activity ever — create the row
        supabase.table("study_streaks").insert({
            "user_id": user_id,
            "current_streak": 1,
            "best_streak": 1,
            "last_activity_date": str(today),
        }).execute()
    else:
        last_date_str = streak_row.get("last_activity_date")
        last_date = date.fromisoformat(last_date_str) if last_date_str else None

        current_streak = streak_row.get("current_streak", 0)
        best_streak = streak_row.get("best_streak", 0)

        if last_date == today:
            # Already recorded today — no streak change
            return
        elif last_date == yesterday:
            current_streak += 1
        else:
            current_streak = 1

        if current_streak > best_streak:
            best_streak = current_streak

        supabase.table("study_streaks").update({
            "current_streak": current_streak,
            "best_streak": best_streak,
            "last_activity_date": str(today),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("user_id", user_id).execute()

    logger.info("Updated streak for user %s", user_id)


def record_note_generated(user_id: str, topic: str) -> None:
    """Record that a note was generated."""
    record_activity(user_id, "note_generated", topic)


def record_flashcard_review(user_id: str, topic: str, correct: bool) -> None:
    """Record a flashcard review."""
    record_activity(user_id, "flashcard_review", topic, {"correct": correct})


def record_quiz_completed(user_id: str, topic: str, score: float, total: int) -> None:
    """Record a completed quiz."""
    record_activity(user_id, "quiz_completed", topic, {"score": score, "total": total})


def record_kb_search(user_id: str, topic: str) -> None:
    """Record a knowledge-base search."""
    record_activity(user_id, "kb_search", topic)


def get_statistics(user_id: str) -> dict[str, Any]:
    """Return aggregated study statistics for *user_id*."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    result = (
        supabase.table("study_activities")
        .select("activity_type, topic, metadata")
        .eq("user_id", user_id)
        .execute()
    )
    rows = result.data or []

    total_notes = 0
    total_flashcards = 0
    flashcard_correct = 0
    total_quizzes = 0
    quiz_score_sum = 0.0
    topics: set[str] = set()

    for row in rows:
        atype = row["activity_type"]
        topic = row.get("topic", "")
        meta = row.get("metadata") or {}

        if topic:
            topics.add(topic)

        if atype == "note_generated":
            total_notes += 1
        elif atype == "flashcard_review":
            total_flashcards += 1
            if meta.get("correct"):
                flashcard_correct += 1
        elif atype == "quiz_completed":
            total_quizzes += 1
            score = meta.get("score")
            total = meta.get("total")
            if score is not None and total:
                quiz_score_sum += (score / total) * 100

    flashcard_accuracy = (
        round(flashcard_correct / total_flashcards * 100, 1) if total_flashcards else 0.0
    )
    quiz_accuracy = round(quiz_score_sum / total_quizzes, 1) if total_quizzes else 0.0

    stats = {
        "total_notes_generated": total_notes,
        "total_flashcards_reviewed": total_flashcards,
        "total_quizzes_completed": total_quizzes,
        "topics_studied": len(topics),
        "flashcard_accuracy": flashcard_accuracy,
        "quiz_accuracy": quiz_accuracy,
    }
    logger.info("Statistics for user %s: %s", user_id, stats)
    return stats


def get_topic_rankings(user_id: str) -> list[dict[str, Any]]:
    """Return topics ranked by activity count with mastery levels."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    result = (
        supabase.table("study_activities")
        .select("topic")
        .eq("user_id", user_id)
        .neq("topic", "")
        .execute()
    )
    rows = result.data or []

    counts: dict[str, int] = {}
    for row in rows:
        topic = row["topic"]
        counts[topic] = counts.get(topic, 0) + 1

    rankings = sorted(
        [
            {
                "topic": topic,
                "activity_count": count,
                "mastery_level": _mastery_level(count),
            }
            for topic, count in counts.items()
        ],
        key=lambda d: d["activity_count"],
        reverse=True,
    )

    logger.info("Topic rankings for user %s: %d topics", user_id, len(rankings))
    return rankings


def get_weekly_summary(user_id: str) -> dict[str, Any]:
    """Return a summary of the current week's activity (Monday to now)."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    monday = _monday_of_current_week()

    result = (
        supabase.table("study_activities")
        .select("activity_type, created_at")
        .eq("user_id", user_id)
        .gte("created_at", monday.isoformat())
        .execute()
    )
    rows = result.data or []

    notes = 0
    flashcards = 0
    quizzes = 0
    active_days: set[str] = set()

    for row in rows:
        atype = row["activity_type"]
        created = row.get("created_at", "")
        if created:
            day = created[:10]  # YYYY-MM-DD
            active_days.add(day)
        if atype == "note_generated":
            notes += 1
        elif atype == "flashcard_review":
            flashcards += 1
        elif atype == "quiz_completed":
            quizzes += 1

    summary = {
        "notes_generated": notes,
        "flashcards_reviewed": flashcards,
        "quizzes_completed": quizzes,
        "active_days": len(active_days),
    }
    logger.info("Weekly summary for user %s: %s", user_id, summary)
    return summary


def get_recent_activities(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return the most recent activities for *user_id*."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    result = (
        supabase.table("study_activities")
        .select("activity_type, topic, metadata, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = result.data or []

    activities = []
    for row in rows:
        atype = row["activity_type"]
        topic = row.get("topic", "")
        meta = row.get("metadata")
        activities.append({
            "summary": _activity_summary(atype, topic, meta),
            "timestamp": row.get("created_at", ""),
            "activity_type": atype,
            "topic": topic,
        })

    logger.info("Fetched %d recent activities for user %s", len(activities), user_id)
    return activities


def reset_progress(user_id: str) -> None:
    """Delete all activities and streak data for *user_id*."""
    from src.db.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    supabase.table("study_activities").delete().eq("user_id", user_id).execute()
    supabase.table("study_streaks").delete().eq("user_id", user_id).execute()

    logger.info("Reset all progress for user %s", user_id)
