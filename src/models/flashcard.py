"""
Flashcard data models for EduNotes Study Features
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum
import uuid


class Difficulty(str, Enum):
    """Difficulty levels for flashcards"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Flashcard(BaseModel):
    """Single flashcard model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    front: str = Field(..., description="Question or term (front of card)")
    back: str = Field(..., description="Answer or definition (back of card)")
    topic: str = Field(..., description="Topic/subject this card belongs to")
    difficulty: Difficulty = Field(default=Difficulty.MEDIUM, description="Card difficulty level")
    created_at: datetime = Field(default_factory=datetime.now)
    last_reviewed: Optional[datetime] = Field(default=None)
    review_count: int = Field(default=0)
    correct_count: int = Field(default=0)
    tags: List[str] = Field(default_factory=list)

    def get_accuracy(self) -> float:
        """Calculate accuracy percentage"""
        if self.review_count == 0:
            return 0.0
        return (self.correct_count / self.review_count) * 100

    def mark_reviewed(self, correct: bool):
        """Mark the card as reviewed"""
        self.review_count += 1
        if correct:
            self.correct_count += 1
        self.last_reviewed = datetime.now()


class FlashcardSet(BaseModel):
    """Collection of flashcards for a topic"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Name of the flashcard set")
    description: Optional[str] = Field(default=None)
    topic: str = Field(..., description="Main topic for this set")
    cards: List[Flashcard] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    source_note_id: Optional[str] = Field(default=None, description="ID of the note this set was generated from")

    def add_card(self, card: Flashcard):
        """Add a flashcard to the set"""
        self.cards.append(card)
        self.updated_at = datetime.now()

    def remove_card(self, card_id: str) -> bool:
        """Remove a flashcard by ID"""
        original_count = len(self.cards)
        self.cards = [c for c in self.cards if c.id != card_id]
        if len(self.cards) < original_count:
            self.updated_at = datetime.now()
            return True
        return False

    def get_card(self, card_id: str) -> Optional[Flashcard]:
        """Get a card by ID"""
        for card in self.cards:
            if card.id == card_id:
                return card
        return None

    def get_cards_by_difficulty(self, difficulty: Difficulty) -> List[Flashcard]:
        """Filter cards by difficulty"""
        return [c for c in self.cards if c.difficulty == difficulty]

    def get_unreviewed_cards(self) -> List[Flashcard]:
        """Get cards that haven't been reviewed yet"""
        return [c for c in self.cards if c.review_count == 0]

    def get_cards_needing_review(self, accuracy_threshold: float = 70.0) -> List[Flashcard]:
        """Get cards with accuracy below threshold"""
        return [c for c in self.cards if c.review_count > 0 and c.get_accuracy() < accuracy_threshold]

    def get_statistics(self) -> dict:
        """Get statistics for the flashcard set"""
        total = len(self.cards)
        if total == 0:
            return {
                "total_cards": 0,
                "reviewed_cards": 0,
                "average_accuracy": 0.0,
                "cards_by_difficulty": {"easy": 0, "medium": 0, "hard": 0}
            }

        reviewed = [c for c in self.cards if c.review_count > 0]
        total_accuracy = sum(c.get_accuracy() for c in reviewed) if reviewed else 0

        return {
            "total_cards": total,
            "reviewed_cards": len(reviewed),
            "average_accuracy": total_accuracy / len(reviewed) if reviewed else 0.0,
            "cards_by_difficulty": {
                "easy": len([c for c in self.cards if c.difficulty == Difficulty.EASY]),
                "medium": len([c for c in self.cards if c.difficulty == Difficulty.MEDIUM]),
                "hard": len([c for c in self.cards if c.difficulty == Difficulty.HARD])
            }
        }


# API Request/Response models
class CreateFlashcardRequest(BaseModel):
    """Request to create flashcards from content"""
    content: str = Field(..., description="Source content to generate flashcards from")
    topic: str = Field(..., description="Topic name for the flashcard set")
    num_cards: int = Field(default=10, ge=1, le=30, description="Number of flashcards to generate")
    set_name: Optional[str] = Field(default=None, description="Custom name for the flashcard set")


class CreateFlashcardResponse(BaseModel):
    """Response after creating flashcards"""
    success: bool
    flashcard_set: Optional[FlashcardSet] = None
    message: str
    error: Optional[str] = None


class ReviewFlashcardRequest(BaseModel):
    """Request to record a flashcard review"""
    set_id: str = Field(..., description="Flashcard set ID")
    card_id: str = Field(..., description="Flashcard ID")
    correct: bool = Field(..., description="Whether the answer was correct")


class ReviewFlashcardResponse(BaseModel):
    """Response after recording a review"""
    success: bool
    card: Optional[Flashcard] = None
    message: str


class ListFlashcardSetsResponse(BaseModel):
    """Response listing all flashcard sets"""
    success: bool
    sets: List[dict]  # Simplified set info without full cards
    total_count: int
