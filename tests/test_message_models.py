"""Role-consistency tests for multi-turn Function Calling messages."""

import pytest
from pydantic import ValidationError

from execsql_agent.models import LLMMessage, ToolCallRequest


def test_assistant_accepts_tool_calls_without_content() -> None:
    call = ToolCallRequest(id="call_1", name="execute_sql", arguments={"sql": "SELECT 1"})

    message = LLMMessage(role="assistant", content=None, tool_calls=[call])

    assert message.tool_calls == [call]


def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        LLMMessage(role="tool", content="observation")


@pytest.mark.parametrize("role", ["system", "user"])
def test_system_and_user_cannot_contain_tool_metadata(role: str) -> None:
    call = ToolCallRequest(id="call_1", name="inspect_schema", arguments={})

    with pytest.raises(ValidationError, match="tool metadata"):
        LLMMessage(role=role, content="content", tool_calls=[call])


def test_empty_assistant_message_is_invalid() -> None:
    with pytest.raises(ValidationError, match="content or tool_calls"):
        LLMMessage(role="assistant", content=None)
