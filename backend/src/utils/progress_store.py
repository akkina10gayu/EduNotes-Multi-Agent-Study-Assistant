"""
Progress Storage Utility for EduNotes
Handles persistence of user study progress
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, date

from config import settings
from src.models.progress import (
    UserProgress, StudyActivity, ActivityType,
    TopicProgress, DailyProgress, StudyStreak
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProgressStore:
    """
    Manages user progress storage and retrieval.
    Stores progress data as JSON files.
    """

    def __init__(self, storage_path: Path = None):
        """
        Initialize the progress store.

        Args:
            storage_path: Path to store progress data
        """
        self.storage_path = storage_path or settings.PROGRESS_STORAGE
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ProgressStore initialized at: {self.storage_path}")

    def _get_progress_file(self, user_id: str = "default") -> Path:
        """Get the file path for user progress"""
        return self.storage_path / f"{user_id}_progress.json"

    def load_progress(self, user_id: str = "default") -> UserProgress:
        """
        Load user progress from storage.

        Args:
            user_id: User identifier (default for single-user mode)

        Returns:
            UserProgress object (creates new if not exists)
        """
        try:
            progress_file = self._get_progress_file(user_id)

            if not progress_file.exists():
                logger.info(f"Creating new progress for user: {user_id}")
                return UserProgress(user_id=user_id)

            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Parse dates
            if isinstance(data.get('created_at'), str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            if isinstance(data.get('updated_at'), str):
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])

            # Parse topic progress
            topic_progress = {}
            for topic, tp_data in data.get('topic_progress', {}).items():
                if tp_data.get('last_studied') and isinstance(tp_data['last_studied'], str):
                    tp_data['last_studied'] = datetime.fromisoformat(tp_data['last_studied'])
                if tp_data.get('first_studied') and isinstance(tp_data['first_studied'], str):
                    tp_data['first_studied'] = datetime.fromisoformat(tp_data['first_studied'])
                topic_progress[topic] = TopicProgress(**tp_data)
            data['topic_progress'] = topic_progress

            # Parse daily progress
            daily_progress = {}
            for day_str, dp_data in data.get('daily_progress', {}).items():
                if isinstance(dp_data.get('date'), str):
                    dp_data['date'] = date.fromisoformat(dp_data['date'])

                # Parse activities within daily progress
                activities = []
                for act_data in dp_data.get('activities', []):
                    if isinstance(act_data.get('timestamp'), str):
                        act_data['timestamp'] = datetime.fromisoformat(act_data['timestamp'])
                    if isinstance(act_data.get('activity_type'), str):
                        act_data['activity_type'] = ActivityType(act_data['activity_type'])
                    activities.append(StudyActivity(**act_data))
                dp_data['activities'] = activities

                daily_progress[day_str] = DailyProgress(**dp_data)
            data['daily_progress'] = daily_progress

            # Parse streak
            streak_data = data.get('streak', {})
            if streak_data.get('last_study_date') and isinstance(streak_data['last_study_date'], str):
                streak_data['last_study_date'] = date.fromisoformat(streak_data['last_study_date'])
            if streak_data.get('streak_start_date') and isinstance(streak_data['streak_start_date'], str):
                streak_data['streak_start_date'] = date.fromisoformat(streak_data['streak_start_date'])
            data['streak'] = StudyStreak(**streak_data) if streak_data else StudyStreak()

            # Parse recent activities
            recent_activities = []
            for act_data in data.get('recent_activities', []):
                if isinstance(act_data.get('timestamp'), str):
                    act_data['timestamp'] = datetime.fromisoformat(act_data['timestamp'])
                if isinstance(act_data.get('activity_type'), str):
                    act_data['activity_type'] = ActivityType(act_data['activity_type'])
                recent_activities.append(StudyActivity(**act_data))
            data['recent_activities'] = recent_activities

            return UserProgress(**data)

        except Exception as e:
            logger.error(f"Error loading progress: {e}")
            return UserProgress(user_id=user_id)

    def save_progress(self, progress: UserProgress) -> bool:
        """
        Save user progress to storage.

        Args:
            progress: UserProgress object to save

        Returns:
            True if successful
        """
        try:
            progress_file = self._get_progress_file(progress.user_id)
            progress.updated_at = datetime.now()

            # Convert to dict for JSON serialization
            data = progress.model_dump()

            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved progress for user: {progress.user_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving progress: {e}")
            return False

    def record_activity(
        self,
        activity_type: ActivityType,
        topic: str,
        details: Dict[str, Any] = None,
        user_id: str = "default"
    ) -> UserProgress:
        """
        Record a study activity and update progress.

        Args:
            activity_type: Type of activity
            topic: Topic related to the activity
            details: Additional activity details
            user_id: User identifier

        Returns:
            Updated UserProgress
        """
        try:
            # Load current progress
            progress = self.load_progress(user_id)

            # Create activity
            activity = StudyActivity(
                activity_type=activity_type,
                topic=topic,
                details=details or {}
            )

            # Record the activity
            progress.record_activity(activity)

            # Save updated progress
            self.save_progress(progress)

            logger.info(f"Recorded activity: {activity_type.value} for topic: {topic}")
            return progress

        except Exception as e:
            logger.error(f"Error recording activity: {e}")
            return self.load_progress(user_id)

    def record_note_generated(self, topic: str, user_id: str = "default") -> UserProgress:
        """Record that a note was generated"""
        return self.record_activity(
            ActivityType.NOTE_GENERATED,
            topic,
            {"action": "generated"},
            user_id
        )

    def record_flashcard_review(
        self,
        topic: str,
        correct: bool,
        user_id: str = "default"
    ) -> UserProgress:
        """Record a flashcard review"""
        return self.record_activity(
            ActivityType.FLASHCARD_REVIEWED,
            topic,
            {"correct": correct},
            user_id
        )

    def record_quiz_completed(
        self,
        topic: str,
        score: float,
        questions_answered: int,
        questions_correct: int,
        user_id: str = "default"
    ) -> UserProgress:
        """Record a completed quiz"""
        progress = self.load_progress(user_id)

        # Update quiz-specific stats
        progress.total_quiz_questions_answered += questions_answered
        progress.total_quiz_questions_correct += questions_correct

        # Record the activity
        activity = StudyActivity(
            activity_type=ActivityType.QUIZ_COMPLETED,
            topic=topic,
            details={
                "score": score,
                "questions_answered": questions_answered,
                "questions_correct": questions_correct
            }
        )
        progress.record_activity(activity)

        # Save
        self.save_progress(progress)

        logger.info(f"Recorded quiz completion: {topic} (score: {score}%)")
        return progress

    def record_kb_search(self, query: str, user_id: str = "default") -> UserProgress:
        """Record a knowledge base search"""
        return self.record_activity(
            ActivityType.KB_SEARCHED,
            "search",
            {"query": query},
            user_id
        )

    def get_statistics(self, user_id: str = "default") -> Dict[str, Any]:
        """Get user statistics"""
        progress = self.load_progress(user_id)
        return progress.get_overall_stats()

    def get_topic_rankings(self, user_id: str = "default") -> list:
        """Get topic rankings by mastery"""
        progress = self.load_progress(user_id)
        return progress.get_topic_rankings()

    def get_weekly_summary(self, user_id: str = "default") -> Dict[str, Any]:
        """Get weekly summary"""
        progress = self.load_progress(user_id)
        return progress.get_weekly_summary()

    def get_recent_activities(self, user_id: str = "default", limit: int = 10) -> list:
        """Get recent activities"""
        progress = self.load_progress(user_id)
        activities = progress.recent_activities[:limit]
        return [
            {
                "id": a.id,
                "type": a.activity_type.value,
                "topic": a.topic,
                "summary": a.to_summary(),
                "timestamp": str(a.timestamp),
                "details": a.details
            }
            for a in activities
        ]

    def reset_progress(self, user_id: str = "default") -> bool:
        """Reset all progress for a user"""
        try:
            progress_file = self._get_progress_file(user_id)
            if progress_file.exists():
                progress_file.unlink()

            logger.info(f"Reset progress for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error resetting progress: {e}")
            return False


# Singleton instance
_progress_store = None


def get_progress_store() -> ProgressStore:
    """Get or create the progress store singleton"""
    global _progress_store
    if _progress_store is None:
        _progress_store = ProgressStore()
    return _progress_store
