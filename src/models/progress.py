"""
Progress tracking data models for EduNotes Study Features
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum
import uuid


class ActivityType(str, Enum):
    """Types of study activities"""
    NOTE_GENERATED = "note_generated"
    FLASHCARD_REVIEWED = "flashcard_reviewed"
    QUIZ_COMPLETED = "quiz_completed"
    KB_SEARCHED = "kb_searched"
    TOPIC_STUDIED = "topic_studied"


class StudyActivity(BaseModel):
    """Record of a single study activity"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    activity_type: ActivityType
    topic: str = Field(default="general")
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_summary(self) -> str:
        """Get a human-readable summary of the activity"""
        type_labels = {
            ActivityType.NOTE_GENERATED: "Generated notes",
            ActivityType.FLASHCARD_REVIEWED: "Reviewed flashcard",
            ActivityType.QUIZ_COMPLETED: "Completed quiz",
            ActivityType.KB_SEARCHED: "Searched KB",
            ActivityType.TOPIC_STUDIED: "Studied topic"
        }
        return f"{type_labels.get(self.activity_type, 'Activity')}: {self.topic}"


class TopicProgress(BaseModel):
    """Progress tracking for a specific topic"""
    topic: str
    notes_generated: int = Field(default=0)
    flashcards_reviewed: int = Field(default=0)
    flashcards_correct: int = Field(default=0)
    quizzes_completed: int = Field(default=0)
    quiz_average_score: float = Field(default=0.0)
    total_study_time_minutes: int = Field(default=0)
    last_studied: Optional[datetime] = None
    first_studied: Optional[datetime] = None

    def get_flashcard_accuracy(self) -> float:
        """Calculate flashcard accuracy percentage"""
        if self.flashcards_reviewed == 0:
            return 0.0
        return (self.flashcards_correct / self.flashcards_reviewed) * 100

    def get_mastery_level(self) -> str:
        """Calculate mastery level based on progress"""
        score = 0

        # Notes contribution (max 20 points)
        score += min(self.notes_generated * 5, 20)

        # Flashcard accuracy contribution (max 40 points)
        if self.flashcards_reviewed > 0:
            accuracy = self.get_flashcard_accuracy()
            score += min(accuracy * 0.4, 40)

        # Quiz performance contribution (max 40 points)
        if self.quizzes_completed > 0:
            score += min(self.quiz_average_score * 0.4, 40)

        if score >= 80:
            return "Expert"
        elif score >= 60:
            return "Advanced"
        elif score >= 40:
            return "Intermediate"
        elif score >= 20:
            return "Beginner"
        else:
            return "Novice"


class DailyProgress(BaseModel):
    """Progress for a single day"""
    date: date
    activities: List[StudyActivity] = Field(default_factory=list)
    topics_studied: List[str] = Field(default_factory=list)
    notes_count: int = Field(default=0)
    flashcards_reviewed: int = Field(default=0)
    quizzes_completed: int = Field(default=0)
    study_time_minutes: int = Field(default=0)

    def add_activity(self, activity: StudyActivity):
        """Add an activity to the day's record"""
        self.activities.append(activity)

        if activity.topic and activity.topic not in self.topics_studied:
            self.topics_studied.append(activity.topic)

        if activity.activity_type == ActivityType.NOTE_GENERATED:
            self.notes_count += 1
        elif activity.activity_type == ActivityType.FLASHCARD_REVIEWED:
            self.flashcards_reviewed += 1
        elif activity.activity_type == ActivityType.QUIZ_COMPLETED:
            self.quizzes_completed += 1


class StudyStreak(BaseModel):
    """Track study streaks"""
    current_streak: int = Field(default=0)
    longest_streak: int = Field(default=0)
    last_study_date: Optional[date] = None
    streak_start_date: Optional[date] = None

    def update(self, study_date: date):
        """Update streak based on new study activity"""
        today = date.today()

        if self.last_study_date is None:
            # First study activity
            self.current_streak = 1
            self.streak_start_date = study_date
        elif study_date == self.last_study_date:
            # Same day, no change
            pass
        elif (study_date - self.last_study_date).days == 1:
            # Consecutive day
            self.current_streak += 1
        elif (study_date - self.last_study_date).days > 1:
            # Streak broken
            self.current_streak = 1
            self.streak_start_date = study_date

        self.last_study_date = study_date

        # Update longest streak
        if self.current_streak > self.longest_streak:
            self.longest_streak = self.current_streak


class UserProgress(BaseModel):
    """Overall user progress tracking"""
    user_id: str = Field(default="default")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Overall statistics
    total_notes_generated: int = Field(default=0)
    total_flashcards_reviewed: int = Field(default=0)
    total_flashcards_correct: int = Field(default=0)
    total_quizzes_completed: int = Field(default=0)
    total_quiz_questions_answered: int = Field(default=0)
    total_quiz_questions_correct: int = Field(default=0)

    # Progress by topic
    topic_progress: Dict[str, TopicProgress] = Field(default_factory=dict)

    # Daily progress
    daily_progress: Dict[str, DailyProgress] = Field(default_factory=dict)

    # Streaks
    streak: StudyStreak = Field(default_factory=StudyStreak)

    # Recent activities
    recent_activities: List[StudyActivity] = Field(default_factory=list)

    def record_activity(self, activity: StudyActivity):
        """Record a new study activity"""
        self.updated_at = datetime.now()

        # Add to recent activities (keep last 50)
        self.recent_activities.insert(0, activity)
        self.recent_activities = self.recent_activities[:50]

        # Update daily progress
        today_str = str(date.today())
        if today_str not in self.daily_progress:
            self.daily_progress[today_str] = DailyProgress(date=date.today())
        self.daily_progress[today_str].add_activity(activity)

        # Update streak
        self.streak.update(date.today())

        # Update topic progress
        if activity.topic:
            if activity.topic not in self.topic_progress:
                self.topic_progress[activity.topic] = TopicProgress(topic=activity.topic)

            tp = self.topic_progress[activity.topic]
            now = datetime.now()

            if tp.first_studied is None:
                tp.first_studied = now
            tp.last_studied = now

            if activity.activity_type == ActivityType.NOTE_GENERATED:
                tp.notes_generated += 1
                self.total_notes_generated += 1
            elif activity.activity_type == ActivityType.FLASHCARD_REVIEWED:
                tp.flashcards_reviewed += 1
                self.total_flashcards_reviewed += 1
                if activity.details.get('correct', False):
                    tp.flashcards_correct += 1
                    self.total_flashcards_correct += 1
            elif activity.activity_type == ActivityType.QUIZ_COMPLETED:
                tp.quizzes_completed += 1
                self.total_quizzes_completed += 1
                score = activity.details.get('score', 0)
                # Update running average
                if tp.quizzes_completed == 1:
                    tp.quiz_average_score = score
                else:
                    tp.quiz_average_score = ((tp.quiz_average_score * (tp.quizzes_completed - 1)) + score) / tp.quizzes_completed

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        flashcard_accuracy = 0.0
        if self.total_flashcards_reviewed > 0:
            flashcard_accuracy = (self.total_flashcards_correct / self.total_flashcards_reviewed) * 100

        quiz_accuracy = 0.0
        if self.total_quiz_questions_answered > 0:
            quiz_accuracy = (self.total_quiz_questions_correct / self.total_quiz_questions_answered) * 100

        return {
            "total_notes_generated": self.total_notes_generated,
            "total_flashcards_reviewed": self.total_flashcards_reviewed,
            "flashcard_accuracy": round(flashcard_accuracy, 1),
            "total_quizzes_completed": self.total_quizzes_completed,
            "quiz_accuracy": round(quiz_accuracy, 1),
            "topics_studied": len(self.topic_progress),
            "current_streak": self.streak.current_streak,
            "longest_streak": self.streak.longest_streak,
            "days_active": len(self.daily_progress)
        }

    def get_topic_rankings(self) -> List[Dict[str, Any]]:
        """Get topics ranked by mastery level"""
        rankings = []
        for topic, progress in self.topic_progress.items():
            rankings.append({
                "topic": topic,
                "mastery_level": progress.get_mastery_level(),
                "notes_generated": progress.notes_generated,
                "flashcard_accuracy": round(progress.get_flashcard_accuracy(), 1),
                "quiz_average": round(progress.quiz_average_score, 1),
                "last_studied": str(progress.last_studied) if progress.last_studied else None
            })

        # Sort by activity (most active first)
        rankings.sort(key=lambda x: x['notes_generated'] + x.get('quiz_average', 0), reverse=True)
        return rankings

    def get_weekly_summary(self) -> Dict[str, Any]:
        """Get summary for the last 7 days"""
        from datetime import timedelta

        today = date.today()
        week_start = today - timedelta(days=6)

        notes = 0
        flashcards = 0
        quizzes = 0
        active_days = 0
        topics = set()

        for i in range(7):
            day = week_start + timedelta(days=i)
            day_str = str(day)
            if day_str in self.daily_progress:
                dp = self.daily_progress[day_str]
                notes += dp.notes_count
                flashcards += dp.flashcards_reviewed
                quizzes += dp.quizzes_completed
                active_days += 1
                topics.update(dp.topics_studied)

        return {
            "period": f"{week_start} to {today}",
            "notes_generated": notes,
            "flashcards_reviewed": flashcards,
            "quizzes_completed": quizzes,
            "active_days": active_days,
            "topics_studied": list(topics)
        }


# API Request/Response models
class GetProgressResponse(BaseModel):
    """Response for progress retrieval"""
    success: bool
    overall_stats: Dict[str, Any]
    topic_rankings: List[Dict[str, Any]]
    weekly_summary: Dict[str, Any]
    recent_activities: List[Dict[str, Any]]
    streak: Dict[str, Any]


class RecordActivityRequest(BaseModel):
    """Request to record a study activity"""
    activity_type: ActivityType
    topic: str
    details: Dict[str, Any] = Field(default_factory=dict)


class RecordActivityResponse(BaseModel):
    """Response after recording an activity"""
    success: bool
    message: str
    current_streak: int
