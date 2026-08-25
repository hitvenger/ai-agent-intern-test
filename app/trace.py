from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from app.models import AgentTrace


class TraceLogger:
    """
    Structured execution telemetry logger for the Aster & Row agent.
    Provides complete observability into retrieval, tool execution, privacy checks, and handoff decisions.
    """

    def __init__(self):
        pass

    def create_trace(
        self,
        session_id: str,
        user_query: str,
        intent: Optional[str] = None,
        order_id: Optional[str] = None,
        tool_called: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        retrieved_citations: Optional[List[str]] = None,
        retrieved_scores: Optional[List[float]] = None,
        conflict_detected: bool = False,
        conflict_details: Optional[str] = None,
        handoff_recommended: bool = False,
        handoff_reason: Optional[str] = None,
        generation_mode: str = "deterministic"
    ) -> AgentTrace:
        return AgentTrace(
            session_id=session_id,
            user_query=user_query,
            intent=intent,
            order_id_extracted=order_id,
            tool_called=tool_called,
            tool_args=tool_args,
            tool_result=tool_result,
            retrieved_citations=retrieved_citations or [],
            retrieved_scores=retrieved_scores or [],
            conflict_detected=conflict_detected,
            conflict_details=conflict_details,
            handoff_recommended=handoff_recommended,
            handoff_reason=handoff_reason,
            generation_mode=generation_mode
        )

    @staticmethod
    def format_trace_for_display(trace: AgentTrace) -> str:
        """Renders trace in a readable format for CLI/debugging."""
        lines = [
            f"[Trace] Session: {trace.session_id}",
            f"[Trace] Query: '{trace.user_query}'",
            f"[Trace] Intent: {trace.intent or 'N/A'}"
        ]
        if trace.tool_called:
            lines.append(f"[Trace] Tool Executed: {trace.tool_called}({json.dumps(trace.tool_args or {})})")
            if trace.tool_result:
                lines.append(f"[Trace] Tool Result Status: {trace.tool_result.get('status', 'OK')}")
        if trace.retrieved_citations:
            lines.append(f"[Trace] Sources: {', '.join(trace.retrieved_citations)}")
        if trace.conflict_detected:
            lines.append(f"[Trace] Conflict Alert: {trace.conflict_details}")
        if trace.handoff_recommended:
            lines.append(f"[Trace] Handoff Recommended: YES (Reason: {trace.handoff_reason})")
        return "\n".join(lines)
