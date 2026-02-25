"""
Flashcard Generator for EduNotes
Generates flashcards from text content using LLM
"""
import re
from typing import List, Optional
from datetime import datetime

from src.models.flashcard import Flashcard, FlashcardSet, Difficulty
from src.utils.llm_client import get_llm_client
from src.db.flashcard_store import get_flashcard_store
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FlashcardGenerator:
    """
    Generates flashcards from educational content using LLM.
    Parses LLM output into structured Flashcard objects.
    """

    def __init__(self):
        """Initialize the flashcard generator"""
        self.llm_client = get_llm_client()
        self.store = get_flashcard_store()

    def generate_flashcards(
        self,
        content: str,
        topic: str,
        num_cards: int = 10,
        set_name: Optional[str] = None
    ) -> Optional[FlashcardSet]:
        """
        Generate a set of flashcards from content.

        Args:
            content: The source text to generate flashcards from
            topic: The topic for the flashcard set
            num_cards: Number of flashcards to generate (default: 10)
            set_name: Optional custom name for the set

        Returns:
            FlashcardSet if successful, None otherwise
        """
        try:
            logger.info(f"Generating {num_cards} flashcards for topic: {topic}")

            # Generate flashcards using LLM
            raw_output = self._generate_raw_flashcards(content, num_cards)

            if not raw_output:
                logger.error("Failed to generate flashcards from LLM")
                return None

            # Parse the raw output into Flashcard objects
            cards = self._parse_flashcards(raw_output, topic)

            if not cards:
                logger.error("Failed to parse flashcards from LLM output")
                return None

            # Create the flashcard set
            flashcard_set = FlashcardSet(
                name=set_name or f"{topic} Flashcards",
                description=f"Flashcards generated from {topic} content",
                topic=topic,
                cards=cards
            )

            logger.info(f"Created flashcard set with {len(cards)} cards")
            return flashcard_set

        except Exception as e:
            logger.error(f"Error generating flashcards: {e}")
            return None

    def _generate_raw_flashcards(self, content: str, num_cards: int) -> Optional[str]:
        """
        Generate raw flashcard text using the LLM.

        Args:
            content: Source content
            num_cards: Number of cards to generate

        Returns:
            Raw LLM output string or None
        """
        try:
            # Check if using local mode
            if self.llm_client.is_local_mode():
                # Fall back to rule-based generation for local mode
                return self._generate_local_flashcards(content, num_cards)

            # Use LLM client's flashcard generation
            result = self.llm_client.generate_flashcards(content, num_cards)

            if result:
                return result

            # If LLM failed, try local fallback
            logger.warning("LLM flashcard generation failed, using local fallback")
            return self._generate_local_flashcards(content, num_cards)

        except Exception as e:
            logger.error(f"Error in raw flashcard generation: {e}")
            return self._generate_local_flashcards(content, num_cards)

    def _generate_local_flashcards(self, content: str, num_cards: int) -> str:
        """
        Generate flashcards using rule-based extraction (fallback for local mode).

        Args:
            content: Source content
            num_cards: Number of cards to generate

        Returns:
            Formatted flashcard string
        """
        logger.info("Using local rule-based flashcard generation")
        cards = []

        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Look for definition patterns
        definition_patterns = [
            r'([A-Z][^.]+?)\s+(?:is|are|refers to|means|describes)\s+([^.]+)',
            r'([A-Z][A-Za-z\s]+?):\s*([^.]+)',
            r'The\s+([A-Z][A-Za-z\s]+?)\s+(?:is|are)\s+([^.]+)'
        ]

        # Extract definitions
        for sentence in sentences:
            for pattern in definition_patterns:
                match = re.search(pattern, sentence)
                if match and len(cards) < num_cards:
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    if len(term) > 3 and len(definition) > 10:
                        cards.append(f"Q: What is {term}?\nA: {definition}")
                        break

        # If we don't have enough, create from key sentences
        if len(cards) < num_cards:
            for sentence in sentences:
                if len(cards) >= num_cards:
                    break
                # Look for sentences with key terms
                if any(kw in sentence.lower() for kw in ['important', 'key', 'main', 'primary', 'essential']):
                    cards.append(f"Q: Explain: {sentence[:50]}...\nA: {sentence}")

        # Fill remaining with generic questions from content
        remaining = num_cards - len(cards)
        if remaining > 0:
            for i, sentence in enumerate(sentences[:remaining]):
                if len(sentence) > 30:
                    # Create a fill-in-the-blank style card
                    words = sentence.split()
                    if len(words) > 5:
                        key_word = words[len(words) // 2]
                        question = sentence.replace(key_word, "___")
                        cards.append(f"Q: Fill in the blank: {question}\nA: {key_word}")

        return "\n\n".join(cards)

    def _parse_flashcards(self, raw_output: str, topic: str) -> List[Flashcard]:
        """
        Parse raw LLM output into Flashcard objects.

        Args:
            raw_output: Raw text output from LLM
            topic: Topic for the cards

        Returns:
            List of Flashcard objects
        """
        cards = []

        try:
            # Split by common delimiters between cards
            card_blocks = re.split(r'\n\n+|\n---\n|(?=Q:|(?=\d+\.))', raw_output)

            for block in card_blocks:
                block = block.strip()
                if not block:
                    continue

                # Try to extract Q and A
                card = self._parse_single_card(block, topic)
                if card:
                    cards.append(card)

        except Exception as e:
            logger.error(f"Error parsing flashcards: {e}")

        return cards

    def _parse_single_card(self, block: str, topic: str) -> Optional[Flashcard]:
        """
        Parse a single card block into a Flashcard object.

        Args:
            block: Text block containing one flashcard
            topic: Topic for the card

        Returns:
            Flashcard object or None
        """
        try:
            front = None
            back = None

            # Pattern 1: Q: ... A: ...
            q_match = re.search(r'Q:\s*(.+?)(?=\nA:|$)', block, re.DOTALL | re.IGNORECASE)
            a_match = re.search(r'A:\s*(.+?)(?=\n\n|$)', block, re.DOTALL | re.IGNORECASE)

            if q_match and a_match:
                front = q_match.group(1).strip()
                back = a_match.group(1).strip()

            # Pattern 2: Question: ... Answer: ...
            if not front or not back:
                q_match = re.search(r'Question:\s*(.+?)(?=\nAnswer:|$)', block, re.DOTALL | re.IGNORECASE)
                a_match = re.search(r'Answer:\s*(.+?)(?=\n\n|$)', block, re.DOTALL | re.IGNORECASE)
                if q_match and a_match:
                    front = q_match.group(1).strip()
                    back = a_match.group(1).strip()

            # Pattern 3: Front: ... Back: ...
            if not front or not back:
                f_match = re.search(r'Front:\s*(.+?)(?=\nBack:|$)', block, re.DOTALL | re.IGNORECASE)
                b_match = re.search(r'Back:\s*(.+?)(?=\n\n|$)', block, re.DOTALL | re.IGNORECASE)
                if f_match and b_match:
                    front = f_match.group(1).strip()
                    back = b_match.group(1).strip()

            # Pattern 4: Numbered format "1. Question\nAnswer"
            if not front or not back:
                numbered_match = re.search(r'^\d+\.\s*(.+?)\n(.+?)$', block, re.DOTALL)
                if numbered_match:
                    front = numbered_match.group(1).strip()
                    back = numbered_match.group(2).strip()

            # Validate we have both front and back
            if front and back and len(front) > 3 and len(back) > 3:
                # Determine difficulty based on answer length
                difficulty = self._estimate_difficulty(front, back)

                return Flashcard(
                    front=front,
                    back=back,
                    topic=topic,
                    difficulty=difficulty
                )

        except Exception as e:
            logger.debug(f"Error parsing card block: {e}")

        return None

    def _estimate_difficulty(self, question: str, answer: str) -> Difficulty:
        """
        Estimate the difficulty of a flashcard based on content.

        Args:
            question: The question text
            answer: The answer text

        Returns:
            Difficulty level
        """
        # Simple heuristics for difficulty
        combined_length = len(question) + len(answer)

        if combined_length < 100:
            return Difficulty.EASY
        elif combined_length < 250:
            return Difficulty.MEDIUM
        else:
            return Difficulty.HARD

    def generate_from_notes(
        self,
        notes: str,
        topic: str,
        num_cards: int = 10
    ) -> Optional[FlashcardSet]:
        """
        Convenience method to generate flashcards from study notes.

        Args:
            notes: Generated study notes
            topic: Topic name
            num_cards: Number of cards to generate

        Returns:
            FlashcardSet if successful
        """
        return self.generate_flashcards(
            content=notes,
            topic=topic,
            num_cards=num_cards,
            set_name=f"{topic} Study Cards"
        )


# Singleton instance
_flashcard_generator = None


def get_flashcard_generator() -> FlashcardGenerator:
    """Get or create the flashcard generator singleton"""
    global _flashcard_generator
    if _flashcard_generator is None:
        _flashcard_generator = FlashcardGenerator()
    return _flashcard_generator
