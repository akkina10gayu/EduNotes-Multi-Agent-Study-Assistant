"""
Persistent storage for chat conversations.
Stores sessions as JSON files in data/conversations/.
"""
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from src.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class ConversationStore:
    """JSON file-based conversation storage."""

    def __init__(self):
        self.storage_dir = Path(settings.CONVERSATION_STORAGE).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filepath(self, session_id: str) -> Path:
        """Resolve a session file path, rejecting path traversal attempts."""
        if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id:
            raise ValueError(f"Invalid session ID: {session_id}")
        filepath = (self.storage_dir / f"{session_id}.json").resolve()
        if not str(filepath).startswith(str(self.storage_dir)):
            raise ValueError(f"Path traversal detected: {session_id}")
        return filepath

    def save_session(self, session: Dict[str, Any]) -> bool:
        """Save a chat session to disk."""
        try:
            session["updated_at"] = datetime.now().isoformat()
            filepath = self._safe_filepath(session["id"])
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session.get('id')}: {e}")
            return False

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a chat session from disk."""
        try:
            filepath = self._safe_filepath(session_id)
            if not filepath.exists():
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions (summary only, sorted by most recent)."""
        sessions = []
        try:
            json_files = list(self.storage_dir.glob("*.json"))
            for filepath in sorted(json_files, key=os.path.getmtime, reverse=True):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append({
                        "id": data.get("id", filepath.stem),
                        "title": data.get("title", "Untitled"),
                        "mode": data.get("mode", "chat"),
                        "message_count": len(data.get("messages", [])),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                    })
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        try:
            filepath = self._safe_filepath(session_id)
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def export_session_markdown(self, session_id: str) -> Optional[str]:
        """Export a session as a markdown document."""
        session = self.load_session(session_id)
        if not session:
            return None

        lines = [f"# {session.get('title', 'Chat Session')}"]
        lines.append(
            f"*Mode: {session.get('mode', 'chat')} | "
            f"Date: {session.get('created_at', '')[:10]}*\n"
        )

        for msg in session.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"**You:** {content}\n")
            else:
                lines.append(f"**AI:** {content}\n")

        return "\n".join(lines)


# Singleton
_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get or create the conversation store singleton."""
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store
