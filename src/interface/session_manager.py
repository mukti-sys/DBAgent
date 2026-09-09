"""
Session manager — handles persistence, resumption, and switching of conversation sessions.
Enables continuing from previous sessions with full conversation memory, filter state,
and database connections restored.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agent.conversation_state import ConversationState, QueryContext

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_DIR = Path("config/sessions")


class SessionManager:
    """Manages session persistence to disk and state restoration."""

    def __init__(self, sessions_dir: Path | str = DEFAULT_SESSIONS_DIR):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_session_id() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_path(self, session_id: str) -> Path:
        clean_id = Path(session_id).name
        return self.sessions_dir / f"{clean_id}.json"

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions, ordered by last updated descending."""
        sessions = []
        for file in self.sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id", file.stem),
                        "name": data.get("name") or file.stem,
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "db_path": data.get("db_path"),
                        "dialect": data.get("dialect", "sqlite"),
                        "model_name": data.get("model_name"),
                        "query_count": len(data.get("history", [])),
                        "last_question": (
                            data.get("history", [])[-1].get("question", "")
                            if data.get("history") else ""
                        ),
                    })
            except Exception as e:
                logger.warning(f"Failed to read session file {file}: {e}")

        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def save_session(
        self,
        session_id: str,
        context: dict[str, Any],
        name: str | None = None,
        conversation_state: ConversationState | None = None,
    ) -> Path:
        """Save a complete session snapshot to disk."""
        path = self._get_path(session_id)
        now_iso = datetime.now().isoformat()

        created_at = now_iso
        existing_name = None
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    created_at = old_data.get("created_at", now_iso)
                    existing_name = old_data.get("name")
            except Exception:
                pass

        session_name = name or existing_name or context.get("session_name") or session_id

        conversation_turns = []
        session_filters = {}
        if conversation_state is not None:
            session_filters = conversation_state.session_filters
            for turn in conversation_state._history:
                if isinstance(turn, QueryContext):
                    conversation_turns.append(asdict(turn))
                elif isinstance(turn, dict):
                    conversation_turns.append(turn)

        history_clean = []
        for h in context.get("history", []):
            history_clean.append({
                "question": h.get("question", ""),
                "sql": h.get("sql"),
                "row_count": h.get("row_count", 0),
                "success": h.get("success", False),
                "timestamp": h.get("timestamp", now_iso),
            })

        data = {
            "session_id": session_id,
            "name": session_name,
            "created_at": created_at,
            "updated_at": now_iso,
            "db_path": context.get("db_path"),
            "dialect": context.get("dialect", "sqlite"),
            "model_name": context.get("model_name"),
            "provider_name": context.get("provider_name"),
            "provider_key": context.get("provider_key"),
            "base_url": context.get("base_url"),
            "history": history_clean,
            "conversation_turns": conversation_turns,
            "session_filters": session_filters,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return path

    def load_session(self, identifier: str) -> dict[str, Any] | None:
        """Load session data by session ID, prefix, or name."""
        clean_id = identifier.strip()
        direct_path = self._get_path(clean_id)
        if direct_path.exists():
            try:
                with open(direct_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading session {direct_path}: {e}")
                return None

        for file in self.sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if (
                        file.stem == clean_id
                        or file.stem.startswith(clean_id)
                        or data.get("session_id", "").startswith(clean_id)
                        or (data.get("name") and data["name"].lower() == clean_id.lower())
                    ):
                        return data
            except Exception:
                continue

        return None

    def delete_session(self, identifier: str) -> bool:
        """Delete a saved session."""
        session_data = self.load_session(identifier)
        if not session_data:
            return False
        session_id = session_data.get("session_id")
        if not session_id:
            return False
        path = self._get_path(session_id)
        if path.exists():
            try:
                path.unlink()
                return True
            except Exception as e:
                logger.error(f"Error deleting session {path}: {e}")
                return False
        return False

    def restore_into_context(
        self,
        session_data: dict[str, Any],
        context: dict[str, Any],
        conversation_state: ConversationState | None = None,
    ) -> dict[str, Any]:
        """Restore loaded session data into context and conversation state."""
        context["session_id"] = session_data["session_id"]
        context["session_name"] = session_data.get("name", session_data["session_id"])
        context["history"] = session_data.get("history", [])

        if session_data.get("dialect"):
            context["dialect"] = session_data["dialect"]

        if session_data.get("model_name"):
            context["model_name"] = session_data["model_name"]
        if session_data.get("provider_name"):
            context["provider_name"] = session_data["provider_name"]
        if session_data.get("provider_key"):
            context["provider_key"] = session_data["provider_key"]
        if session_data.get("base_url"):
            context["base_url"] = session_data["base_url"]

        if conversation_state is not None:
            conversation_state._history.clear()
            conversation_state._session_filters.clear()

            turns = session_data.get("conversation_turns", [])
            for turn in turns:
                if isinstance(turn, dict):
                    qc = QueryContext(
                        question=turn.get("question", ""),
                        sql=turn.get("sql"),
                        filters=turn.get("filters", {}),
                        tables_used=turn.get("tables_used", []),
                        columns_referenced=turn.get("columns_referenced", []),
                        aggregation=turn.get("aggregation"),
                        time_range=turn.get("time_range"),
                        result_summary=turn.get("result_summary") or f"{turn.get('row_count', 0)} rows",
                    )
                    conversation_state._history.append(qc)

            session_filters = session_data.get("session_filters", {})
            if isinstance(session_filters, dict):
                conversation_state._session_filters.update(session_filters)

        return session_data
