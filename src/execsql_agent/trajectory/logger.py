"""Append reloadable agent trajectories to JSONL."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import cast
from uuid import uuid4

from execsql_agent.models import (
    CompletionKind,
    FunctionCallingAgentResult,
    FunctionCallingStep,
    LLMMessage,
    PipelineAgentResult,
    PipelineStep,
    PipelineTerminationReason,
    Trajectory,
)


class TrajectoryLogger:
    """Persist one complete validated Trajectory per UTF-8 JSONL line."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def log(self, trajectory: Trajectory) -> Path:
        """Validate, append, and return the trajectory file path."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(trajectory.model_dump_json())
            stream.write("\n")
        return self.output_path.resolve()

    def read_recent(self, session_id: str, *, limit: int = 5) -> list[Trajectory]:
        """Reload the newest validated trajectories for one isolated session."""

        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not self.output_path.is_file():
            return []
        recent: deque[Trajectory] = deque(maxlen=limit)
        with self.output_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    trajectory = Trajectory.model_validate_json(line)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid trajectory JSONL at line {line_number}: {error}"
                    ) from error
                if trajectory.session_id == session_id:
                    recent.append(trajectory)
        return list(recent)

    @staticmethod
    def from_function_result(
        result: FunctionCallingAgentResult,
        *,
        llm_backend: str,
        run_mode: str,
    ) -> Trajectory:
        """Build a trajectory without exposing provider credentials."""

        assistant_tool_calls = [
            call for step in result.steps for call in step.tool_calls
        ]
        tool_observations = [
            message for step in result.steps for message in step.tool_messages
        ]
        message_history = [LLMMessage(role="user", content=result.question)]
        for step in result.steps:
            if step.llm_response is not None:
                message_history.append(
                    LLMMessage(
                        role="assistant",
                        content=step.llm_response.final_answer,
                        tool_calls=step.llm_response.tool_calls,
                    )
                )
            message_history.extend(
                message.to_llm_message() for message in step.tool_messages
            )
        validations = [
            validation for step in result.steps for validation in step.validations
        ]
        tool_results = [
            tool_result for step in result.steps for tool_result in step.tool_results
        ]
        return Trajectory(
            trajectory_id=result.trajectory_id,
            session_id=result.session_id,
            agent_mode="function-calling",
            question=result.question,
            database_id=result.database_id,
            schema_summary=result.schema_summary,
            llm_backend=llm_backend,
            run_mode=run_mode,
            steps=cast(list[PipelineStep | FunctionCallingStep], result.steps),
            final_answer=result.final_answer,
            final_sql=result.final_sql,
            execution_result=result.execution_result,
            protocol_completed=result.protocol_completed,
            execution_success=result.execution_success,
            answer_grounded=result.answer_grounded,
            result_correct=result.result_correct,
            completion_kind=result.completion_kind,
            termination_reason=result.termination_reason.value,
            total_llm_turns=result.total_llm_turns,
            tool_call_count=result.tool_call_count,
            sql_execution_count=result.sql_execution_count,
            total_tool_calls=result.tool_call_count,
            total_sql_executions=result.sql_execution_count,
            total_duration_ms=result.total_duration_ms,
            message_history=message_history,
            assistant_tool_calls=assistant_tool_calls,
            tool_observations=tool_observations,
            invalid_tool_calls=sum(
                validation.error_code
                in {"unknown_tool", "invalid_json_arguments", "invalid_arguments"}
                for validation in validations
            ),
            repeated_tool_calls=sum(validation.duplicate for validation in validations),
            unsafe_sql_count=sum(
                tool_result.error_code == "unsafe_sql" for tool_result in tool_results
            ),
        )

    @staticmethod
    def from_pipeline_result(
        result: PipelineAgentResult,
        *,
        llm_backend: str,
        run_mode: str,
    ) -> Trajectory:
        """Represent the Pipeline baseline in the same JSONL envelope."""

        completed = result.termination_reason is PipelineTerminationReason.COMPLETED
        execution_count = sum(
            1 for step in result.steps if step.execution and step.execution.executed
        )
        duration_ms = sum(
            step.execution.duration_ms
            for step in result.steps
            if step.execution is not None
        )
        return Trajectory(
            trajectory_id=str(uuid4()),
            agent_mode="pipeline",
            question=result.question,
            database_id=result.database_id,
            schema_summary=result.schema_summary,
            llm_backend=llm_backend,
            run_mode=run_mode,
            steps=cast(list[PipelineStep | FunctionCallingStep], result.steps),
            final_sql=result.final_sql,
            execution_result=result.execution_result,
            protocol_completed=result.protocol_completed,
            execution_success=result.execution_success,
            answer_grounded=result.answer_grounded,
            result_correct=result.result_correct,
            completion_kind=(
                CompletionKind.TOOL_GROUNDED_ANSWER
                if completed
                else CompletionKind.TERMINATED
            ),
            termination_reason=result.termination_reason.value,
            total_llm_turns=len(result.steps),
            tool_call_count=0,
            sql_execution_count=execution_count,
            total_tool_calls=0,
            total_sql_executions=execution_count,
            total_duration_ms=duration_ms,
            message_history=[LLMMessage(role="user", content=result.question)],
            unsafe_sql_count=sum(
                step.safety is not None and not step.safety.safe
                for step in result.steps
            ),
        )
