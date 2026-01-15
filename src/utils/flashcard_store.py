"""
Flashcard Storage Utility for EduNotes
Handles persistence of flashcard sets to JSON files
"""
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from config import settings
from src.models.flashcard import Flashcard, FlashcardSet, Difficulty
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FlashcardStore:
    """
    Manages flashcard storage and retrieval.
    Stores flashcard sets as JSON files in the configured directory.
    """

    def __init__(self, storage_path: Path = None):
        """
        Initialize the flashcard store.

        Args:
            storage_path: Path to store flashcards. Uses settings.FLASHCARD_STORAGE if not provided.
        """
        self.storage_path = storage_path or settings.FLASHCARD_STORAGE
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self._ensure_index()
        logger.info(f"FlashcardStore initialized at: {self.storage_path}")

    def _ensure_index(self):
        """Ensure the index file exists"""
        if not self.index_file.exists():
            self._save_index({})

    def _load_index(self) -> Dict[str, Any]:
        """Load the index of all flashcard sets"""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return {}

    def _save_index(self, index: Dict[str, Any]):
        """Save the index to file"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def _get_set_path(self, set_id: str) -> Path:
        """Get the file path for a flashcard set"""
        return self.storage_path / f"{set_id}.json"

    def save_set(self, flashcard_set: FlashcardSet) -> bool:
        """
        Save a flashcard set to storage.

        Args:
            flashcard_set: The FlashcardSet to save

        Returns:
            True if successful, False otherwise
        """
        try:
            set_path = self._get_set_path(flashcard_set.id)

            # Convert to dict for JSON serialization
            set_data = flashcard_set.model_dump()

            # Save the set file
            with open(set_path, 'w', encoding='utf-8') as f:
                json.dump(set_data, f, indent=2, default=str)

            # Update index
            index = self._load_index()
            index[flashcard_set.id] = {
                "name": flashcard_set.name,
                "topic": flashcard_set.topic,
                "card_count": len(flashcard_set.cards),
                "created_at": str(flashcard_set.created_at),
                "updated_at": str(flashcard_set.updated_at)
            }
            self._save_index(index)

            logger.info(f"Saved flashcard set: {flashcard_set.name} ({flashcard_set.id})")
            return True

        except Exception as e:
            logger.error(f"Error saving flashcard set: {e}")
            return False

    def load_set(self, set_id: str) -> Optional[FlashcardSet]:
        """
        Load a flashcard set from storage.

        Args:
            set_id: The ID of the flashcard set to load

        Returns:
            FlashcardSet if found, None otherwise
        """
        try:
            set_path = self._get_set_path(set_id)

            if not set_path.exists():
                logger.warning(f"Flashcard set not found: {set_id}")
                return None

            with open(set_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Parse dates
            data['created_at'] = datetime.fromisoformat(data['created_at']) if isinstance(data['created_at'], str) else data['created_at']
            data['updated_at'] = datetime.fromisoformat(data['updated_at']) if isinstance(data['updated_at'], str) else data['updated_at']

            # Parse cards
            cards = []
            for card_data in data.get('cards', []):
                if isinstance(card_data.get('created_at'), str):
                    card_data['created_at'] = datetime.fromisoformat(card_data['created_at'])
                if card_data.get('last_reviewed') and isinstance(card_data['last_reviewed'], str):
                    card_data['last_reviewed'] = datetime.fromisoformat(card_data['last_reviewed'])
                if isinstance(card_data.get('difficulty'), str):
                    card_data['difficulty'] = Difficulty(card_data['difficulty'])
                cards.append(Flashcard(**card_data))

            data['cards'] = cards

            return FlashcardSet(**data)

        except Exception as e:
            logger.error(f"Error loading flashcard set {set_id}: {e}")
            return None

    def delete_set(self, set_id: str) -> bool:
        """
        Delete a flashcard set from storage.

        Args:
            set_id: The ID of the flashcard set to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            set_path = self._get_set_path(set_id)

            if set_path.exists():
                set_path.unlink()

            # Update index
            index = self._load_index()
            if set_id in index:
                del index[set_id]
                self._save_index(index)

            logger.info(f"Deleted flashcard set: {set_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting flashcard set {set_id}: {e}")
            return False

    def list_sets(self) -> List[Dict[str, Any]]:
        """
        List all flashcard sets (summary info only).

        Returns:
            List of dictionaries with set summaries
        """
        try:
            index = self._load_index()
            sets = []

            for set_id, info in index.items():
                sets.append({
                    "id": set_id,
                    **info
                })

            # Sort by updated_at (most recent first)
            sets.sort(key=lambda x: x.get('updated_at', ''), reverse=True)

            return sets

        except Exception as e:
            logger.error(f"Error listing flashcard sets: {e}")
            return []

    def get_sets_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get all flashcard sets for a specific topic.

        Args:
            topic: The topic to filter by

        Returns:
            List of set summaries matching the topic
        """
        all_sets = self.list_sets()
        return [s for s in all_sets if s.get('topic', '').lower() == topic.lower()]

    def update_card_review(self, set_id: str, card_id: str, correct: bool) -> Optional[Flashcard]:
        """
        Update a card's review status.

        Args:
            set_id: The flashcard set ID
            card_id: The card ID
            correct: Whether the answer was correct

        Returns:
            Updated Flashcard if successful, None otherwise
        """
        try:
            flashcard_set = self.load_set(set_id)
            if not flashcard_set:
                return None

            card = flashcard_set.get_card(card_id)
            if not card:
                logger.warning(f"Card not found: {card_id}")
                return None

            # Update the card
            card.mark_reviewed(correct)

            # Save the updated set
            self.save_set(flashcard_set)

            logger.info(f"Updated card review: {card_id} (correct: {correct})")
            return card

        except Exception as e:
            logger.error(f"Error updating card review: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics for all flashcard sets.

        Returns:
            Dictionary with aggregate statistics
        """
        try:
            sets = self.list_sets()
            total_cards = sum(s.get('card_count', 0) for s in sets)
            topics = list(set(s.get('topic', 'unknown') for s in sets))

            return {
                "total_sets": len(sets),
                "total_cards": total_cards,
                "topics": topics,
                "topic_count": len(topics)
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "total_sets": 0,
                "total_cards": 0,
                "topics": [],
                "topic_count": 0
            }


# Singleton instance
_flashcard_store = None


def get_flashcard_store() -> FlashcardStore:
    """Get or create the flashcard store singleton"""
    global _flashcard_store
    if _flashcard_store is None:
        _flashcard_store = FlashcardStore()
    return _flashcard_store
