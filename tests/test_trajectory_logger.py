"""JSONL trajectory round-trip tests."""

from pathlib import Path

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import (
    FunctionCallingStep,
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
    Trajectory,
)
from execsql_agent.trajectory.logger import TrajectoryLogger


def test_function_trajectory_round_trips_through_jsonl(
    demo_db: Path, tmp_path: Path
) -> None:
    path = tmp_path / "trajectories.jsonl"
    fake = FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="execute",
                        name="execute_sql",
                        arguments={"sql": "SELECT COUNT(*) FROM customers"},
                    )
                ],
                response_mode=ResponseMode.NATIVE_TOOL_CALLS,
            ),
            LLMResponse(
                final_answer="共有 10 名客户。",
                response_mode=ResponseMode.PLAIN_FINAL,
            ),
        ]
    )
    logger = TrajectoryLogger(path)

    result = FunctionCallingAgent(
        demo_db, fake, trajectory_logger=logger
    ).run("客户数量")

    raw_line = path.read_text(encoding="utf-8").strip()
    trajectory = Trajectory.model_validate_json(raw_line)
    assert trajectory.trajectory_id == result.trajectory_id
    assert trajectory.run_mode == "deterministic/mock"
    assert trajectory.steps[0].step_type == "function_calling"
    assert trajectory.protocol_completed is True
    assert "API_KEY" not in raw_line
    assert result.trajectory_path == str(path.resolve())


def test_unresolved_completion_guard_is_recorded_in_trajectory(
    demo_db: Path, tmp_path: Path
) -> None:
    path = tmp_path / "unresolved.jsonl"
    fake = FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="inspect",
                        name="inspect_schema",
                        arguments={"table_names": ["customers"]},
                    )
                ],
                response_mode=ResponseMode.NATIVE_TOOL_CALLS,
            ),
            LLMResponse(
                final_answer="I should query next.",
                response_mode=ResponseMode.PLAIN_FINAL,
            ),
            LLMResponse(
                final_answer="I still did not query.",
                response_mode=ResponseMode.PLAIN_FINAL,
            ),
        ]
    )

    result = FunctionCallingAgent(
        demo_db,
        fake,
        trajectory_logger=TrajectoryLogger(path),
    ).run("客户数量")

    trajectory = Trajectory.model_validate_json(path.read_text(encoding="utf-8"))
    assert result.termination_reason.value == "unresolved_completion"
    assert trajectory.termination_reason == "unresolved_completion"
    assert trajectory.protocol_completed is False
    assert trajectory.total_llm_turns == 3
    guard_text = "The task has not yet produced or successfully executed SQL."
    guard_step = trajectory.steps[1]
    final_step = trajectory.steps[-1]
    assert isinstance(guard_step, FunctionCallingStep)
    assert isinstance(final_step, FunctionCallingStep)
    assert guard_step.completion_guard_triggered is True
    assert guard_text in (guard_step.completion_guard_instruction or "")
    assert final_step.completion_guard_triggered is False
    assert [
        message.content
        for message in trajectory.message_history
        if message.role == "user"
    ] == ["客户数量"]
    assert all(
        guard_text not in (message.content or "")
        for message in trajectory.message_history
    )
