"""Conversation-memory tests using real SQLite tool executions."""

from pathlib import Path

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import LLMResponse, ResponseMode, ToolCallRequest
from execsql_agent.trajectory.logger import TrajectoryLogger


def _execute(call_id: str, sql: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            ToolCallRequest(
                id=call_id,
                name="execute_sql",
                arguments={"sql": sql},
            )
        ],
        response_mode=ResponseMode.NATIVE_TOOL_CALLS,
    )


def _final(answer: str) -> LLMResponse:
    return LLMResponse(
        final_answer=answer,
        response_mode=ResponseMode.PLAIN_FINAL,
    )


def test_follow_up_region_uses_recent_session_context(
    demo_db: Path, tmp_path: Path
) -> None:
    first_sql = "SELECT COUNT(*) AS customer_count FROM customers WHERE city = '北京'"
    follow_up_sql = "SELECT COUNT(*) AS customer_count FROM customers WHERE city = '上海'"
    first_fake = FakeLLMClient(
        [_execute("beijing", first_sql), _final("北京有 1 名客户。")]
    )
    logger = TrajectoryLogger(tmp_path / "sessions.jsonl")
    first_agent = FunctionCallingAgent(
        demo_db, first_fake, trajectory_logger=logger
    )

    first = first_agent.run("北京有多少客户？", session_id="region-session")
    second_fake = FakeLLMClient(
        [_execute("shanghai", follow_up_sql), _final("上海也有 1 名客户。")]
    )
    second_agent = FunctionCallingAgent(
        demo_db, second_fake, trajectory_logger=logger
    )
    second = second_agent.run("那上海呢？", session_id="region-session")

    assert first.execution_result is not None
    assert second.execution_result is not None
    assert second.execution_result.rows == [[1]]
    follow_up_prompt = second_fake.requests[0].messages[0].content or ""
    assert "北京有多少客户？" in follow_up_prompt
    assert first_sql in follow_up_prompt
    assert "execute_sql" in follow_up_prompt
    assert "returned_rows=1" in follow_up_prompt
    assert "北京有 1 名客户。" in follow_up_prompt
    assert [
        turn.question
        for turn in second_agent.memory_store.recent("region-session")
    ] == [
        "北京有多少客户？",
        "那上海呢？",
    ]


def test_follow_up_changes_previous_filter_condition(demo_db: Path) -> None:
    first_sql = (
        "SELECT product_name FROM products WHERE category = '电脑配件' "
        "ORDER BY product_id LIMIT 5"
    )
    changed_sql = (
        "SELECT product_name FROM products WHERE category = '办公家具' "
        "ORDER BY product_id LIMIT 5"
    )
    fake = FakeLLMClient(
        [
            _execute("accessories", first_sql),
            _final("已列出电脑配件。"),
            _execute("furniture", changed_sql),
            _final("已改为办公家具。"),
        ]
    )
    agent = FunctionCallingAgent(demo_db, fake)

    agent.run("列出前五个电脑配件。", session_id="filter-session")
    changed = agent.run("换成办公家具。", session_id="filter-session")

    assert changed.final_sql == changed_sql
    assert changed.execution_result is not None
    assert changed.execution_result.rows == [["人体工学椅"], ["升降桌"]]
    follow_up_prompt = fake.requests[2].messages[0].content or ""
    assert "列出前五个电脑配件。" in follow_up_prompt
    assert "category = '电脑配件'" in follow_up_prompt
    assert "已列出电脑配件。" in follow_up_prompt


def test_sessions_are_isolated_and_memory_keeps_only_five_turns(
    demo_db: Path,
) -> None:
    responses = [
        response
        for index in range(6)
        for response in (
            _final(f"会话 A 回答 {index}"),
            _final(f"会话 A 仍未解决 {index}"),
        )
    ]
    responses.extend([_final("会话 B 回答"), _final("会话 B 仍未解决")])
    fake = FakeLLMClient(responses)
    agent = FunctionCallingAgent(demo_db, fake)

    for index in range(6):
        agent.run(f"会话 A 问题 {index}", session_id="session-a")
    agent.run("会话 B 独立问题", session_id="session-b")

    session_a = agent.memory_store.recent("session-a")
    session_b = agent.memory_store.recent("session-b")
    assert len(session_a) == 5
    assert [turn.question for turn in session_a] == [
        "会话 A 问题 1",
        "会话 A 问题 2",
        "会话 A 问题 3",
        "会话 A 问题 4",
        "会话 A 问题 5",
    ]
    assert [turn.question for turn in session_b] == ["会话 B 独立问题"]
    session_b_prompt = fake.requests[12].messages[0].content or ""
    assert "会话 A" not in session_b_prompt

def test_completion_guard_is_not_stored_as_user_session_memory(
    demo_db: Path,
) -> None:
    guard_text = "The task has not yet produced or successfully executed SQL."
    fake = FakeLLMClient(
        [
            _final("I should query the database next."),
            _final("I cannot complete the task."),
        ]
    )
    agent = FunctionCallingAgent(demo_db, fake)

    agent.run("Count the customers.", session_id="guard-session")

    prompt = agent.memory_store.render_prompt("guard-session")
    assert guard_text not in prompt
    assert "Count the customers." in prompt
    assert [
        message.content
        for message in fake.requests[-1].messages
        if message.role == "user"
    ] == ["Count the customers."]
    assert guard_text in (fake.requests[-1].messages[0].content or "")
