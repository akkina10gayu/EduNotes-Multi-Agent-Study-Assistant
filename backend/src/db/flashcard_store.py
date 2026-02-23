"""
Flashcard Storage backed by Supabase for EduNotes v2.
Replaces the JSON-file-based FlashcardStore from v1.
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from src.db.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class FlashcardStore:
    """
    Manages flashcard storage and retrieval using Supabase.

    Tables:
        flashcard_sets  (id, user_id, topic, source_content, card_count, created_at)
        flashcards      (id, set_id, front, back, difficulty, times_reviewed,
                         times_correct, last_reviewed, created_at)
    """

    def __init__(self):
        self._client = get_supabase_client()
        logger.info("FlashcardStore initialized (Supabase-backed)")

    # ------------------------------------------------------------------
    # save_set
    # ------------------------------------------------------------------
    def save_set(
        self,
        user_id: str,
        topic: str,
        cards: List[Dict[str, Any]],
        source_content: str = "",
    ) -> Dict[str, Any]:
        """Save a new flashcard set with its cards.

        Args:
            user_id: Owner of the set.
            topic: Topic / name of the flashcard set.
            cards: List of card dicts, each with at least 'front' and 'back'.
            source_content: Optional source material the cards were generated from.

        Returns:
            Dict matching v1 format: id, name, topic, cards, created_at.
        """
        # 1. Insert the set row
        set_row = (
            self._client.table("flashcard_sets")
            .insert(
                {
                    "user_id": user_id,
                    "topic": topic,
                    "source_content": source_content,
                    "card_count": len(cards),
                }
            )
            .execute()
        )
        set_data = set_row.data[0]
        set_id = set_data["id"]

        # 2. Bulk-insert cards
        card_rows = []
        for card in cards:
            card_rows.append(
                {
                    "set_id": set_id,
                    "front": card.get("front", ""),
                    "back": card.get("back", ""),
                    "difficulty": card.get("difficulty", "medium"),
                }
            )

        inserted_cards = (
            self._client.table("flashcards").insert(card_rows).execute()
        )

        # 3. Update card_count (in case the actual insert count differs)
        actual_count = len(inserted_cards.data)
        if actual_count != set_data["card_count"]:
            self._client.table("flashcard_sets").update(
                {"card_count": actual_count}
            ).eq("id", set_id).execute()

        # 4. Return v1-compatible dict
        return {
            "id": set_id,
            "name": topic,
            "topic": topic,
            "cards": inserted_cards.data,
            "created_at": set_data["created_at"],
        }

    # ------------------------------------------------------------------
    # load_set
    # ------------------------------------------------------------------
    def load_set(
        self, user_id: str, set_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load a flashcard set with all its cards.

        Args:
            user_id: Owner of the set (used for access control).
            set_id: The flashcard set ID.

        Returns:
            Dict with id, name, topic, cards, created_at -- or None.
        """
        set_resp = (
            self._client.table("flashcard_sets")
            .select("*")
            .eq("id", set_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not set_resp.data:
            logger.warning(
                f"Flashcard set not found: {set_id} for user {user_id}"
            )
            return None

        set_data = set_resp.data[0]

        cards_resp = (
            self._client.table("flashcards")
            .select("*")
            .eq("set_id", set_id)
            .execute()
        )

        return {
            "id": set_data["id"],
            "name": set_data["topic"],
            "topic": set_data["topic"],
            "cards": cards_resp.data,
            "created_at": set_data["created_at"],
        }

    # ------------------------------------------------------------------
    # delete_set
    # ------------------------------------------------------------------
    def delete_set(self, user_id: str, set_id: str) -> bool:
        """Delete a flashcard set (cascades to flashcards via FK).

        Args:
            user_id: Owner of the set.
            set_id: The flashcard set ID.

        Returns:
            True if deleted, False otherwise.
        """
        try:
            self._client.table("flashcard_sets").delete().eq(
                "id", set_id
            ).eq("user_id", user_id).execute()
            logger.info(f"Deleted flashcard set: {set_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting flashcard set {set_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # list_sets
    # ------------------------------------------------------------------
    def list_sets(self, user_id: str) -> List[Dict[str, Any]]:
        """List all flashcard sets for a user (summary only).

        Args:
            user_id: Owner of the sets.

        Returns:
            List of dicts: id, name, card_count, created_at.
        """
        resp = (
            self._client.table("flashcard_sets")
            .select("id, topic, card_count, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        return [
            {
                "id": row["id"],
                "name": row["topic"],
                "card_count": row["card_count"],
                "created_at": row["created_at"],
            }
            for row in resp.data
        ]

    # ------------------------------------------------------------------
    # update_card_review
    # ------------------------------------------------------------------
    def update_card_review(
        self,
        user_id: str,
        set_id: str,
        card_id: str,
        correct: bool,
    ) -> Optional[Dict[str, Any]]:
        """Update a card's review statistics.

        Args:
            user_id: Owner of the set.
            set_id: The flashcard set ID.
            card_id: The card ID.
            correct: Whether the answer was correct.

        Returns:
            Updated card data dict, or None on failure.
        """
        # Verify set belongs to user
        set_resp = (
            self._client.table("flashcard_sets")
            .select("id")
            .eq("id", set_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not set_resp.data:
            logger.warning(
                f"Set {set_id} not found for user {user_id}"
            )
            return None

        # Fetch current card to increment counters
        card_resp = (
            self._client.table("flashcards")
            .select("*")
            .eq("id", card_id)
            .eq("set_id", set_id)
            .execute()
        )
        if not card_resp.data:
            logger.warning(f"Card {card_id} not found in set {set_id}")
            return None

        card = card_resp.data[0]
        updates: Dict[str, Any] = {
            "times_reviewed": card["times_reviewed"] + 1,
            "last_reviewed": datetime.now(timezone.utc).isoformat(),
        }
        if correct:
            updates["times_correct"] = card["times_correct"] + 1

        updated = (
            self._client.table("flashcards")
            .update(updates)
            .eq("id", card_id)
            .execute()
        )

        logger.info(
            f"Updated card review: {card_id} (correct: {correct})"
        )
        return updated.data[0] if updated.data else None

    # ------------------------------------------------------------------
    # export_to_anki
    # ------------------------------------------------------------------
    def export_to_anki(self, user_id: str, set_id: str) -> Optional[str]:
        """Export a flashcard set to Anki-compatible tab-separated format.

        Args:
            user_id: Owner of the set.
            set_id: The flashcard set ID.

        Returns:
            Tab-separated text (front\\tback per line), or None.
        """
        data = self.load_set(user_id, set_id)
        if not data:
            logger.error(
                f"Cannot export: flashcard set {set_id} not found"
            )
            return None

        lines = []
        for card in data["cards"]:
            front = card["front"].replace("\t", " ").replace("\n", "<br>")
            back = card["back"].replace("\t", " ").replace("\n", "<br>")
            lines.append(f"{front}\t{back}")

        logger.info(
            f"Exported {len(lines)} cards from set {set_id} to Anki format"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # export_all_to_anki
    # ------------------------------------------------------------------
    def export_all_to_anki(
        self, user_id: str
    ) -> Dict[str, Any]:
        """Export all flashcard sets for a user to Anki format.

        Args:
            user_id: Owner of the sets.

        Returns:
            Dict with 'text' (tab-separated content) and 'total_cards' count.
        """
        sets = self.list_sets(user_id)
        all_lines: List[str] = []

        for s in sets:
            data = self.load_set(user_id, s["id"])
            if not data:
                continue
            for card in data["cards"]:
                front = card["front"].replace("\t", " ").replace("\n", "<br>")
                back = card["back"].replace("\t", " ").replace("\n", "<br>")
                all_lines.append(f"{front}\t{back}")

        total_cards = len(all_lines)
        text = "\n".join(all_lines)
        logger.info(
            f"Exported {total_cards} cards from all sets to Anki format"
        )
        return {"text": text, "total_cards": total_cards}

    # ------------------------------------------------------------------
    # get_sets_by_topic
    # ------------------------------------------------------------------
    def get_sets_by_topic(
        self, user_id: str, topic: str
    ) -> List[Dict[str, Any]]:
        """Get all flashcard sets for a user filtered by topic.

        Args:
            user_id: Owner of the sets.
            topic: Topic to filter by (case-insensitive).

        Returns:
            List of matching set summaries.
        """
        resp = (
            self._client.table("flashcard_sets")
            .select("id, topic, card_count, created_at")
            .eq("user_id", user_id)
            .ilike("topic", topic)
            .order("created_at", desc=True)
            .execute()
        )

        return [
            {
                "id": row["id"],
                "name": row["topic"],
                "card_count": row["card_count"],
                "created_at": row["created_at"],
            }
            for row in resp.data
        ]

    # ------------------------------------------------------------------
    # get_statistics
    # ------------------------------------------------------------------
    def get_statistics(self, user_id: str) -> Dict[str, Any]:
        """Aggregate statistics for all of a user's flashcard data.

        Args:
            user_id: Owner of the sets.

        Returns:
            Dict with total_sets, total_cards, total_reviews, accuracy.
        """
        try:
            sets = self.list_sets(user_id)
            total_sets = len(sets)
            total_cards = sum(s.get("card_count", 0) for s in sets)

            # Aggregate review stats from flashcards table
            set_ids = [s["id"] for s in sets]
            total_reviews = 0
            total_correct = 0

            if set_ids:
                cards_resp = (
                    self._client.table("flashcards")
                    .select("times_reviewed, times_correct")
                    .in_("set_id", set_ids)
                    .execute()
                )
                for card in cards_resp.data:
                    total_reviews += card.get("times_reviewed", 0)
                    total_correct += card.get("times_correct", 0)

            accuracy = (
                round(total_correct / total_reviews, 4)
                if total_reviews > 0
                else 0.0
            )

            return {
                "total_sets": total_sets,
                "total_cards": total_cards,
                "total_reviews": total_reviews,
                "accuracy": accuracy,
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "total_sets": 0,
                "total_cards": 0,
                "total_reviews": 0,
                "accuracy": 0.0,
            }


# Singleton instance
_flashcard_store: Optional[FlashcardStore] = None


def get_flashcard_store() -> FlashcardStore:
    """Get or create the FlashcardStore singleton."""
    global _flashcard_store
    if _flashcard_store is None:
        _flashcard_store = FlashcardStore()
    return _flashcard_store
