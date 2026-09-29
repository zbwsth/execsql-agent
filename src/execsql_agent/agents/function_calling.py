"""Native minimal Function Calling Agent loop."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from execsql_agent.llm.base import LLMClient, LLMClientError
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import (
    CompletionKind,
    ExecutionResult,
    FunctionCallingAgentResult,
    FunctionCallingStep,
    FunctionCallingTerminationReason,
    LLMMessage,
    LLMRequest,
    ToolCallRequest,
    ToolCallResult,
    ToolCallValidation,
    ToolMessage,
)
from execsql_agent.tools.registry import ToolRegistry
from execsql_agent.tools.schema_loader import SchemaLoader
from execsql_agent.trajectory.logger import TrajectoryLogger
from execsql_agent.trajectory.memory import SessionMemoryStore

_COMPLETION_GUARD_MESSAGE = (
    "The task has not yet produced or successfully executed SQL. "
    "If the database can answer the question, continue using the available tools. "
    "If it cannot be completed, explicitly explain why."
)


class FunctionCallingAgent:
    """Let an LLM iteratively select safe allowlisted SQLite tools."""

    def __init__(
        self,
        database_path: str | Path,
        client: LLMClient,
        *,
        registry: ToolRegistry | None = None,
        trajectory_logger: TrajectoryLogger | None = None,
        max_steps: int = 5,
        llm_backend: str | None = None,
        run_mode: str | None = None,
        memory_store: SessionMemoryStore | None = None,
        domain_context: str | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.database_path = Path(database_path)
        self.client = client
        self.registry = registry or ToolRegistry(self.database_path)
        self.trajectory_logger = trajectory_logger
        self.max_steps = max_steps
        self.llm_backend = llm_backend or type(client).__name__
        self.run_mode = run_mode or (
            "deterministic/mock" if isinstance(client, FakeLLMClient) else "live"
        )
        self.memory_store = memory_store or SessionMemoryStore(max_turns=5)
        self.domain_context = domain_context.strip() if domain_context else None
        self._hydrated_sessions: set[str] = set()

    def run(self, question: str, *, session_id: str = "default") -> FunctionCallingAgentResult:
        """Run model turns and ordered tool observations to an explicit terminal state."""

        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id must not be blank")
        self._hydrate_session(session_id)
        started_at = perf_counter()
        trajectory_id = str(uuid4())
        try:
            schema = SchemaLoader(self.database_path).load()
        except (OSError, sqlite3.Error):
            return self._finalize(
                trajectory_id=trajectory_id,
                session_id=session_id,
                question=question,
                database_id=self.database_path.stem,
                schema_summary="",
                steps=[],
                final_answer=None,
                final_sql=None,
                execution_result=None,
                termination_reason=FunctionCallingTerminationReason.UNRECOVERABLE_ERROR,
                tool_call_count=0,
                sql_execution_count=0,
                started_at=started_at,
            )

        memory_context = self.memory_store.render_prompt(session_id)
        system_content = (
            "You are a read-only SQLite analysis agent. Use only the provided tools. "
            "Inspect schema when needed, never invent tool names or fields, and give a "
            "concise final answer only after enough observations. Exploratory SQL is "
            "allowed. Before producing a database-backed final answer, the last "
            "execute_sql call must directly return exactly the complete columns and "
            "complete rows required by the user's question. Do not combine earlier SQL "
            "observations, do not omit required fields, and do not return extra columns "
            "or extra rows. Base the final answer only on the last successful execute_sql "
            "result. After a successful execute_sql result already completely answers "
            "the user question, stop calling tools and provide the final answer; never "
            "repeat the same execute_sql call. For Top-K requests, use the exact "
            "requested LIMIT in that final SQL; "
            "never fetch more rows and truncate them only in natural language."
        )
        if self.domain_context:
            system_content += f"\n\nDomain context:\n{self.domain_context}"
        if memory_context:
            system_content += f"\n\n{memory_context}"
        history = [
            LLMMessage(
                role="system",
                content=system_content,
            ),
            LLMMessage(role="user", content=question),
        ]
        steps: list[FunctionCallingStep] = []
        invalid_signatures: set[str] = set()
        executed_sql: set[str] = set()
        validate_epochs: dict[str, int] = {}
        inspect_error_epochs: dict[str, int] = {}
        context_epoch = 0
        execution_error_epoch = 0
        unsafe_pending = False
        last_execution: ExecutionResult | None = None
        last_successful_execution: ExecutionResult | None = None
        final_sql: str | None = None
        tool_call_count = 0
        sql_execution_count = 0
        completion_guard_used = False
        completion_guard_pending = False

        for turn_index in range(1, self.max_steps + 1):
            turn_started = perf_counter()
            input_summary = [self._message_summary(message) for message in history]
            request_messages = list(history)
            if completion_guard_pending:
                request_messages = self._with_completion_guard(request_messages)
                completion_guard_pending = False
            try:
                response = self.client.complete(
                    LLMRequest(
                        messages=request_messages,
                        tools=self.registry.definitions,
                    )
                )
            except LLMClientError as error:
                steps.append(
                    FunctionCallingStep(
                        turn_index=turn_index,
                        input_message_summary=input_summary,
                        llm_error=error.error,
                        duration_ms=(perf_counter() - turn_started) * 1000,
                    )
                )
                return self._finalize(
                    trajectory_id=trajectory_id,
                    session_id=session_id,
                    question=question,
                    database_id=schema.database_id,
                    schema_summary=schema.summary_text,
                    steps=steps,
                    final_answer=None,
                    final_sql=final_sql,
                    execution_result=last_successful_execution or last_execution,
                    termination_reason=FunctionCallingTerminationReason.MODEL_ERROR,
                    tool_call_count=tool_call_count,
                    sql_execution_count=sql_execution_count,
                    started_at=started_at,
                )

            if not response.tool_calls:
                history.append(
                    LLMMessage(role="assistant", content=response.final_answer)
                )
                unresolved = (
                    final_sql is None and last_successful_execution is None
                )
                guard_triggered = unresolved and not completion_guard_used
                steps.append(
                    FunctionCallingStep(
                        turn_index=turn_index,
                        input_message_summary=input_summary,
                        llm_response=response,
                        completion_guard_triggered=guard_triggered,
                        completion_guard_instruction=(
                            _COMPLETION_GUARD_MESSAGE if guard_triggered else None
                        ),
                        duration_ms=(perf_counter() - turn_started) * 1000,
                    )
                )
                if guard_triggered:
                    completion_guard_used = True
                    completion_guard_pending = True
                    continue
                return self._finalize(
                    trajectory_id=trajectory_id,
                    session_id=session_id,
                    question=question,
                    database_id=schema.database_id,
                    schema_summary=schema.summary_text,
                    steps=steps,
                    final_answer=response.final_answer,
                    final_sql=final_sql,
                    execution_result=last_successful_execution or last_execution,
                    termination_reason=(
                        FunctionCallingTerminationReason.UNRESOLVED_COMPLETION
                        if unresolved
                        else FunctionCallingTerminationReason.COMPLETED
                    ),
                    tool_call_count=tool_call_count,
                    sql_execution_count=sql_execution_count,
                    started_at=started_at,
                )

            history.append(
                LLMMessage(
                    role="assistant",
                    content=response.final_answer,
                    tool_calls=response.tool_calls,
                )
            )
            validations: list[ToolCallValidation] = []
            results: list[ToolCallResult] = []
            tool_messages: list[ToolMessage] = []
            signatures_in_turn: set[str] = set()
            terminal_reason: FunctionCallingTerminationReason | None = None

            for call in response.tool_calls:
                tool_call_count += 1
                signature = self.registry.canonical_signature(call)
                base_validation = self.registry.validate_call(call)
                same_turn_duplicate = signature in signatures_in_turn
                duplicate = same_turn_duplicate
                duplicate_code = "repeated_tool_call"

                if not base_validation.valid:
                    duplicate = duplicate or signature in invalid_signatures
                elif not same_turn_duplicate and call.name == "execute_sql":
                    sql = self._sql_argument(call)
                    safety = self.registry.validator.validate(sql) if sql else None
                    if safety and safety.safe and safety.normalized_sql in executed_sql:
                        duplicate = True
                        duplicate_code = "repeated_sql"
                elif not same_turn_duplicate and call.name == "validate_sql":
                    duplicate = duplicate or validate_epochs.get(signature) == context_epoch
                elif not same_turn_duplicate and call.name == "inspect_schema":
                    duplicate = duplicate or (
                        inspect_error_epochs.get(signature) == execution_error_epoch
                    )

                validation, result = self.registry.dispatch(
                    call, duplicate=duplicate, duplicate_code=duplicate_code
                )
                signatures_in_turn.add(signature)
                validations.append(validation)
                results.append(result)
                tool_message = ToolMessage(
                    tool_call_id=call.id,
                    name=call.name,
                    content=result.model_dump_json(),
                )
                tool_messages.append(tool_message)
                history.append(tool_message.to_llm_message())

                if not base_validation.valid:
                    if duplicate:
                        terminal_reason = (
                            FunctionCallingTerminationReason.REPEATED_TOOL_CALL
                        )
                    else:
                        invalid_signatures.add(signature)
                    if terminal_reason is not None:
                        break
                    continue

                if duplicate:
                    terminal_reason = (
                        FunctionCallingTerminationReason.REPEATED_SQL
                        if duplicate_code == "repeated_sql"
                        else FunctionCallingTerminationReason.REPEATED_TOOL_CALL
                    )
                    break

                if call.name == "inspect_schema":
                    inspect_error_epochs[signature] = execution_error_epoch
                    context_epoch += 1
                elif call.name == "validate_sql":
                    validate_epochs[signature] = context_epoch
                elif call.name == "execute_sql":
                    sql = self._sql_argument(call)
                    if sql is not None:
                        final_sql = sql
                    execution = self.registry.execution_result(result)
                    if execution is not None:
                        last_execution = execution
                        if execution.executed:
                            sql_execution_count += 1
                            safety = self.registry.validator.validate(sql or "")
                            if safety.safe:
                                executed_sql.add(safety.normalized_sql)
                        if execution.execution_success:
                            last_successful_execution = execution
                            context_epoch += 1
                        elif execution.executed:
                            execution_error_epoch += 1
                            context_epoch += 1

                if call.name in {"validate_sql", "execute_sql"}:
                    sql_candidate = self._sql_argument(call)
                    if sql_candidate is not None:
                        final_sql = sql_candidate
                    if result.error_code == "unsafe_sql":
                        if unsafe_pending:
                            terminal_reason = FunctionCallingTerminationReason.UNSAFE_SQL
                            break
                        unsafe_pending = True
                    else:
                        unsafe_pending = False

            steps.append(
                FunctionCallingStep(
                    turn_index=turn_index,
                    input_message_summary=input_summary,
                    llm_response=response,
                    tool_calls=response.tool_calls,
                    validations=validations,
                    tool_results=results,
                    tool_messages=tool_messages,
                    duration_ms=(perf_counter() - turn_started) * 1000,
                )
            )
            if terminal_reason is not None:
                return self._finalize(
                    trajectory_id=trajectory_id,
                    session_id=session_id,
                    question=question,
                    database_id=schema.database_id,
                    schema_summary=schema.summary_text,
                    steps=steps,
                    final_answer=None,
                    final_sql=final_sql,
                    execution_result=last_successful_execution or last_execution,
                    termination_reason=terminal_reason,
                    tool_call_count=tool_call_count,
                    sql_execution_count=sql_execution_count,
                    started_at=started_at,
                )

        return self._finalize(
            trajectory_id=trajectory_id,
            session_id=session_id,
            question=question,
            database_id=schema.database_id,
            schema_summary=schema.summary_text,
            steps=steps,
            final_answer=None,
            final_sql=final_sql,
            execution_result=last_successful_execution or last_execution,
            termination_reason=FunctionCallingTerminationReason.MAX_STEPS_REACHED,
            tool_call_count=tool_call_count,
            sql_execution_count=sql_execution_count,
            started_at=started_at,
        )

    def _finalize(
        self,
        *,
        trajectory_id: str,
        session_id: str,
        question: str,
        database_id: str,
        schema_summary: str,
        steps: list[FunctionCallingStep],
        final_answer: str | None,
        final_sql: str | None,
        execution_result: ExecutionResult | None,
        termination_reason: FunctionCallingTerminationReason,
        tool_call_count: int,
        sql_execution_count: int,
        started_at: float,
    ) -> FunctionCallingAgentResult:
        completed = termination_reason is FunctionCallingTerminationReason.COMPLETED
        execution_success = bool(
            execution_result is not None and execution_result.execution_success
        )
        if completed and execution_success:
            completion_kind = CompletionKind.TOOL_GROUNDED_ANSWER
        elif completed:
            completion_kind = CompletionKind.DIRECT_ANSWER
        else:
            completion_kind = CompletionKind.TERMINATED
        result = FunctionCallingAgentResult(
            trajectory_id=trajectory_id,
            session_id=session_id,
            question=question,
            database_id=database_id,
            schema_summary=schema_summary,
            steps=steps,
            final_answer=final_answer,
            final_sql=final_sql,
            execution_result=execution_result,
            protocol_completed=completed,
            execution_success=execution_success,
            answer_grounded=completed and execution_success,
            result_correct=None,
            completion_kind=completion_kind,
            termination_reason=termination_reason,
            total_llm_turns=len(steps),
            tool_call_count=tool_call_count,
            sql_execution_count=sql_execution_count,
            total_duration_ms=(perf_counter() - started_at) * 1000,
        )
        trajectory = TrajectoryLogger.from_function_result(
            result, llm_backend=self.llm_backend, run_mode=self.run_mode
        )
        self.memory_store.remember(session_id, trajectory)
        if self.trajectory_logger is not None:
            path = self.trajectory_logger.log(trajectory)
            result = result.model_copy(update={"trajectory_path": str(path)})
        return result

    def _hydrate_session(self, session_id: str) -> None:
        """Load a session's recent JSONL trajectories at most once per Agent instance."""

        if session_id in self._hydrated_sessions:
            return
        if self.trajectory_logger is not None and not self.memory_store.recent(session_id):
            for trajectory in self.trajectory_logger.read_recent(session_id, limit=5):
                self.memory_store.remember(session_id, trajectory)
        self._hydrated_sessions.add(session_id)

    @staticmethod
    def _sql_argument(call: ToolCallRequest) -> str | None:
        if call.arguments is None:
            return None
        value = call.arguments.get("sql")
        return value if isinstance(value, str) else None

    @staticmethod
    def _with_completion_guard(history: list[LLMMessage]) -> list[LLMMessage]:
        """Add the one-shot guard only to the next provider request."""

        messages = list(history)
        if not messages or messages[0].role != "system":
            raise ValueError("Function Calling history must start with a system message")
        system_message = messages[0]
        messages[0] = system_message.model_copy(
            update={
                "content": (
                    f"{system_message.content}\n\n"
                    "Temporary harness instruction for this decision only:\n"
                    f"{_COMPLETION_GUARD_MESSAGE}"
                )
            }
        )
        return messages

    @staticmethod
    def _message_summary(message: LLMMessage) -> str:
        content = (message.content or "").replace("\n", " ")[:160]
        calls = ",".join(call.name for call in message.tool_calls)
        suffix = f" tool_calls=[{calls}]" if calls else ""
        call_id = f" tool_call_id={message.tool_call_id}" if message.tool_call_id else ""
        return f"{message.role}: {content}{suffix}{call_id}"
