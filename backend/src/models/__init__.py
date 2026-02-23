"""Data models for EduNotes"""

from src.models.schemas import (
    GenerateNotesRequest, GenerateNotesResponse,
    UpdateKBRequest, UpdateKBResponse,
    SearchKBRequest, SearchKBResponse,
    StatsResponse, HealthResponse
)

from src.models.flashcard import (
    Flashcard, FlashcardSet, Difficulty,
    CreateFlashcardRequest, CreateFlashcardResponse,
    ReviewFlashcardRequest, ReviewFlashcardResponse,
    ListFlashcardSetsResponse
)

from src.models.quiz import (
    Quiz, QuizQuestion, QuizAttempt, QuestionType,
    CreateQuizRequest, CreateQuizResponse,
    SubmitAnswerRequest, SubmitAnswerResponse,
    CompleteQuizRequest, CompleteQuizResponse,
    ListQuizzesResponse
)

from src.models.progress import (
    UserProgress, StudyActivity, ActivityType,
    TopicProgress, DailyProgress, StudyStreak,
    GetProgressResponse, RecordActivityRequest, RecordActivityResponse
)