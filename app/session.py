from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: str  # "user", "assistant"
    content: str
    sources: List[str] = Field(default_factory=list)
    order_id: Optional[str] = None
    handoff: bool = False


class ConversationSession(BaseModel):
    session_id: str
    turns: List[ConversationTurn] = Field(default_factory=list)
    active_order_id: Optional[str] = None
    active_topic: Optional[str] = None

    def add_user_message(self, content: str, detected_order_id: Optional[str] = None) -> None:
        if detected_order_id:
            self.active_order_id = detected_order_id
        self.turns.append(ConversationTurn(role="user", content=content, order_id=self.active_order_id))

    def add_assistant_response(self, content: str, sources: List[str] = [], handoff: bool = False) -> None:
        self.turns.append(ConversationTurn(
            role="assistant",
            content=content,
            sources=sources,
            order_id=self.active_order_id,
            handoff=handoff
        ))

    def get_context_summary(self) -> str:
        """Returns sliding-window summary of recent turns for multi-turn grounding."""
        recent = self.turns[-6:] if len(self.turns) > 6 else self.turns
        lines = []
        for t in recent:
            lines.append(f"{t.role.capitalize()}: {t.content}")
        return "\n".join(lines)


class SessionManager:
    """In-memory session manager for multi-turn conversations."""

    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create(self, session_id: str) -> ConversationSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id=session_id)
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
