"""
Quiz data models for EduNotes Study Features
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class QuestionType(str, Enum):
    """Types of quiz questions"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"


class QuizQuestion(BaseModel):
    """Single quiz question model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., description="The question text")
    question_type: QuestionType = Field(default=QuestionType.MULTIPLE_CHOICE)
    options: List[str] = Field(default_factory=list, description="Answer options for multiple choice")
    correct_answer: str = Field(..., description="The correct answer")
    correct_index: Optional[int] = Field(default=None, description="Index of correct option for MCQ")
    explanation: Optional[str] = Field(default=None, description="Explanation of the correct answer")
    topic: str = Field(default="general")
    difficulty: str = Field(default="medium")

    def check_answer(self, user_answer: str) -> bool:
        """Check if the user's answer is correct"""
        return user_answer.strip().lower() == self.correct_answer.strip().lower()


class QuizAttempt(BaseModel):
    """Record of a quiz attempt"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    quiz_id: str
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    answers: Dict[str, str] = Field(default_factory=dict)  # question_id -> user_answer
    results: Dict[str, bool] = Field(default_factory=dict)  # question_id -> correct
    score: float = Field(default=0.0)
    total_questions: int = Field(default=0)
    correct_count: int = Field(default=0)

    def submit_answer(self, question_id: str, answer: str, correct: bool):
        """Record an answer for a question"""
        self.answers[question_id] = answer
        self.results[question_id] = correct
        if correct:
            self.correct_count += 1

    def complete(self):
        """Mark the quiz attempt as complete"""
        self.completed_at = datetime.now()
        self.total_questions = len(self.answers)
        if self.total_questions > 0:
            self.score = (self.correct_count / self.total_questions) * 100


class Quiz(BaseModel):
    """A quiz containing multiple questions"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Quiz title")
    description: Optional[str] = Field(default=None)
    topic: str = Field(..., description="Topic the quiz covers")
    questions: List[QuizQuestion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    source_content: Optional[str] = Field(default=None, description="Source content the quiz was generated from")
    attempts: List[QuizAttempt] = Field(default_factory=list)

    def add_question(self, question: QuizQuestion):
        """Add a question to the quiz"""
        self.questions.append(question)

    def get_question(self, question_id: str) -> Optional[QuizQuestion]:
        """Get a question by ID"""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get quiz statistics"""
        if not self.attempts:
            return {
                "total_attempts": 0,
                "average_score": 0.0,
                "best_score": 0.0,
                "total_questions": len(self.questions)
            }

        completed = [a for a in self.attempts if a.completed_at]
        scores = [a.score for a in completed]

        return {
            "total_attempts": len(self.attempts),
            "completed_attempts": len(completed),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "best_score": max(scores) if scores else 0.0,
            "total_questions": len(self.questions)
        }

    def start_attempt(self) -> QuizAttempt:
        """Start a new quiz attempt"""
        attempt = QuizAttempt(
            quiz_id=self.id,
            total_questions=len(self.questions)
        )
        self.attempts.append(attempt)
        return attempt


# API Request/Response models
class CreateQuizRequest(BaseModel):
    """Request to create a quiz from content"""
    content: str = Field(..., description="Source content to generate quiz from")
    topic: str = Field(..., description="Topic name for the quiz")
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of questions to generate")
    title: Optional[str] = Field(default=None, description="Custom quiz title")


class CreateQuizResponse(BaseModel):
    """Response after creating a quiz"""
    success: bool
    quiz: Optional[Quiz] = None
    message: str
    error: Optional[str] = None


class SubmitAnswerRequest(BaseModel):
    """Request to submit an answer"""
    quiz_id: str
    attempt_id: str
    question_id: str
    answer: str


class SubmitAnswerResponse(BaseModel):
    """Response after submitting an answer"""
    success: bool
    correct: bool
    correct_answer: str
    explanation: Optional[str] = None
    message: str


class CompleteQuizRequest(BaseModel):
    """Request to complete a quiz attempt"""
    quiz_id: str
    attempt_id: str


class QuestionResult(BaseModel):
    """Detailed result for a single question"""
    question_id: str
    question_text: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    options: List[str]
    explanation: Optional[str] = None


class CompleteQuizResponse(BaseModel):
    """Response after completing a quiz"""
    success: bool
    score: float
    correct_count: int
    total_questions: int
    results: Dict[str, bool]  # Keep for backward compatibility
    detailed_results: List[QuestionResult] = Field(default_factory=list)
    message: str


class ListQuizzesResponse(BaseModel):
    """Response listing all quizzes"""
    success: bool
    quizzes: List[dict]
    total_count: int
