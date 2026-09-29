"""Small bounded session memory derived from existing agent trajectories."""

from __future__ import annotations

from collections import deque
from threading import RLock

from execsql_agent.models import SessionMemoryTurn, ToolCallRequest, Trajectory


class SessionMemoryStore:
    """Keep recent summarized turns isolated by caller-provided session ID."""

    def __init__(self, *, max_turns: int = 5, max_turn_chars: int = 1_200) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if max_turn_chars < 1_000:
            raise ValueError("max_turn_chars must be at least 1000")
        self.max_turns = max_turns
        self.max_turn_chars = max_turn_chars
        self._sessions: dict[str, deque[SessionMemoryTurn]] = {}
        self._lock = RLock()

    def remember(self, session_id: str, trajectory: Trajectory) -> SessionMemoryTurn:
        """Summarize one trajectory and append it to only the selected session."""

        normalized_id = self._validate_session_id(session_id)
        turn = self._summarize(trajectory)
        with self._lock:
            turns = self._sessions.setdefault(
                normalized_id, deque(maxlen=self.max_turns)
            )
            turns.append(turn)
        return turn

    def recent(self, session_id: str) -> list[SessionMemoryTurn]:
        """Return a copy of the oldest-to-newest bounded session history."""

        normalized_id = self._validate_session_id(session_id)
        with self._lock:
            return list(self._sessions.get(normalized_id, ()))

    def render_prompt(self, session_id: str) -> str:
        """Render only compact recent summaries suitable for a system prompt."""

        turns = self.recent(session_id)
        if not turns:
            return ""
        sections = [
            "Recent conversation memory follows. Resolve short follow-ups using this "
            "context, while treating the newest user question as authoritative."
        ]
        for index, turn in enumerate(turns, start=1):
            sql_text = self._clip(" | ".join(turn.generated_sql) or "none", 360)
            call_text = ", ".join(self._render_call(call) for call in turn.tool_calls)
            section = (
                f"Turn {index}:\n"
                f"User question: {self._clip(turn.question, 100)}\n"
                f"Generated SQL: {sql_text}\n"
                f"Tool calls: {self._clip(call_text or 'none', 100)}\n"
                f"Execution: {self._clip(turn.execution_summary, 160)}\n"
                f"Final answer: {self._clip(turn.final_answer or 'none', 120)}"
            )
            sections.append(section[: self.max_turn_chars])
        return "\n\n".join(sections)

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("session_id must not be blank")
        return normalized

    @staticmethod
    def _render_call(call: ToolCallRequest) -> str:
        argument_keys = sorted(call.arguments) if call.arguments is not None else []
        return f"{call.name}({','.join(argument_keys)})"

    @staticmethod
    def _clip(value: str, limit: int) -> str:
        return value if len(value) <= limit else f"{value[: limit - 3]}..."

    @staticmethod
    def _summarize(trajectory: Trajectory) -> SessionMemoryTurn:
        generated_sql: list[str] = []
        for call in trajectory.assistant_tool_calls:
            if call.name not in {"validate_sql", "execute_sql"}:
                continue
            sql = call.arguments.get("sql") if call.arguments is not None else None
            if isinstance(sql, str) and sql not in generated_sql:
                generated_sql.append(sql)
        if trajectory.final_sql and trajectory.final_sql not in generated_sql:
            generated_sql.append(trajectory.final_sql)

        execution = trajectory.execution_result
        if execution is None:
            execution_summary = "not executed"
        elif execution.execution_success:
            sample_rows = execution.rows[:3]
            execution_summary = (
                f"success; returned_rows={execution.returned_row_count}; "
                f"columns={execution.columns}; sample_rows={sample_rows}; "
                f"truncated={execution.truncated}"
            )
        else:
            error = execution.error.message if execution.error is not None else None
            execution_summary = (
                f"failed; executed={execution.executed}; "
                f"error={error or execution.blocked_reason or 'unknown'}"
            )
        return SessionMemoryTurn(
            question=trajectory.question,
            generated_sql=generated_sql,
            tool_calls=trajectory.assistant_tool_calls,
            execution_summary=execution_summary,
            final_answer=trajectory.final_answer,
            termination_reason=trajectory.termination_reason,
        )
