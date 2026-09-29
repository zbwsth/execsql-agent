"""Tests for deterministic and OpenAI-compatible LLM clients."""

import json

import httpx
import pytest

from execsql_agent.llm.base import LLMClientError
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.llm.openai_compatible import (
    OpenAICompatibleLLMClient,
    OpenAICompatibleSettings,
)
from execsql_agent.models import (
    LLMCallConfig,
    LLMError,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
    ToolDefinition,
)


def _settings(base_url: str = "https://llm.example") -> OpenAICompatibleSettings:
    return OpenAICompatibleSettings(api_key="test-key", base_url=base_url, model="test-model")


def _request(
    *,
    tools: list[ToolDefinition] | None = None,
    response_schema: dict[str, object] | None = None,
) -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage(role="user", content="question")],
        tools=tools or [],
        config=LLMCallConfig(response_schema=response_schema),
    )


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="inspect_schema",
        description="Inspect SQLite schema",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )


def _api_response(message: dict[str, object], finish_reason: str = "stop") -> dict[str, object]:
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


def test_fake_llm_returns_queued_responses_and_records_requests() -> None:
    queued = LLMResponse(
        final_answer='{"sql":"SELECT 1"}',
        response_mode=ResponseMode.PLAIN_FINAL,
    )
    client = FakeLLMClient([queued])
    request = _request()

    assert client.complete(request) == queued
    assert client.requests == [request]
    assert client.remaining_responses == 0


def test_fake_llm_queue_exhaustion_is_structured() -> None:
    client = FakeLLMClient([])

    with pytest.raises(LLMClientError) as error_info:
        client.complete(_request())

    assert error_info.value.error.code == "fake_queue_exhausted"
    assert error_info.value.error.retryable is False


def test_fake_llm_can_express_multiple_tool_calls() -> None:
    response = LLMResponse(
        tool_calls=[
            ToolCallRequest(id="one", name="inspect_schema", arguments={}),
            ToolCallRequest(id="two", name="execute_sql", arguments={"sql": "SELECT 1"}),
        ],
        response_mode=ResponseMode.NATIVE_TOOL_CALLS,
    )

    returned = FakeLLMClient([response]).complete(_request())

    assert [call.name for call in returned.tool_calls] == ["inspect_schema", "execute_sql"]


def test_parses_native_single_and_multiple_tool_calls() -> None:
    payload = _api_response(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "inspect_schema", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "execute_sql",
                        "arguments": '{"sql":"SELECT 1"}',
                    },
                },
            ],
        },
        finish_reason="tool_calls",
    )
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    http_client = httpx.Client(transport=transport)
    client = OpenAICompatibleLLMClient(_settings(), http_client=http_client)

    response = client.complete(_request(tools=[_tool()]))

    assert response.response_mode is ResponseMode.NATIVE_TOOL_CALLS
    assert [call.id for call in response.tool_calls] == ["call_1", "call_2"]
    assert response.tool_calls[1].arguments == {"sql": "SELECT 1"}


def test_parses_plain_final_answer_and_builds_endpoint() -> None:
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json=_api_response({"content": "final answer"}))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleLLMClient(
        _settings("https://llm.example/v1"), http_client=http_client
    )

    response = client.complete(_request())

    assert response.final_answer == "final answer"
    assert response.response_mode is ResponseMode.PLAIN_FINAL
    assert seen_urls == ["https://llm.example/v1/chat/completions"]


def test_explicit_decoding_config_overrides_request_defaults() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_api_response({"content": "done"}))

    client = OpenAICompatibleLLMClient(
        _settings(),
        temperature=0.0,
        top_p=0.95,
        max_tokens=2048,
        seed=17,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    client.complete(
        LLMRequest(
            messages=[LLMMessage(role="user", content="question")],
            config=LLMCallConfig(temperature=0.7, max_tokens=64),
        )
    )

    assert bodies[0]["temperature"] == 0.0
    assert bodies[0]["top_p"] == 0.95
    assert bodies[0]["max_tokens"] == 2048
    assert bodies[0]["seed"] == 17


def test_client_reads_required_openai_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, json=_api_response({"content": "ok"})
            )
        )
    )

    client = OpenAICompatibleLLMClient(http_client=http_client)

    assert client.settings.api_key == "env-key"
    assert client.settings.model == "env-model"
    assert client.endpoint == "https://env.example/v1/chat/completions"


def test_retries_without_response_format_when_rejected() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(400, text="unknown response_format: unsupported")
        return httpx.Response(200, json=_api_response({"content": '{"ok":true}'}))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleLLMClient(_settings(), http_client=http_client)

    response = client.complete(_request(response_schema={"type": "object"}))

    assert response.final_answer == '{"ok":true}'
    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1]


def test_uses_json_fallback_when_tools_are_unsupported() -> None:
    bodies: list[dict[str, object]] = []
    fallback_content = json.dumps(
        {
            "type": "tool_calls",
            "tool_calls": [
                {"id": "fallback_1", "name": "inspect_schema", "arguments": {}}
            ],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(400, text="tools are not supported")
        return httpx.Response(200, json=_api_response({"content": fallback_content}))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleLLMClient(_settings(), http_client=http_client)

    response = client.complete(_request(tools=[_tool()]))

    assert response.response_mode is ResponseMode.JSON_FALLBACK
    assert response.tool_calls[0].name == "inspect_schema"
    assert "tools" in bodies[0]
    assert "tools" not in bodies[1]
    assert bodies[1]["messages"][0]["role"] == "system"


def test_rejects_invalid_provider_json() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, text="not-json"))
    client = OpenAICompatibleLLMClient(
        _settings(), http_client=httpx.Client(transport=transport)
    )

    with pytest.raises(LLMClientError) as error_info:
        client.complete(_request())

    assert error_info.value.error.code == "invalid_response"


def test_rejects_incomplete_native_tool_call() -> None:
    payload = _api_response(
        {
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "x"}}
            ],
        }
    )
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    client = OpenAICompatibleLLMClient(
        _settings(), http_client=httpx.Client(transport=transport)
    )

    with pytest.raises(LLMClientError) as error_info:
        client.complete(_request(tools=[_tool()]))

    assert error_info.value.error.code == "invalid_response"


def test_retries_timeout_then_succeeds() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=_api_response({"content": "recovered"}))

    client = OpenAICompatibleLLMClient(
        _settings(),
        max_retries=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.complete(_request()).final_answer == "recovered"
    assert attempts == 2


def test_retryable_http_error_exhaustion_is_structured() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(503, text="temporarily unavailable")
    )
    client = OpenAICompatibleLLMClient(
        _settings(),
        max_retries=1,
        http_client=httpx.Client(transport=transport),
    )

    with pytest.raises(LLMClientError) as error_info:
        client.complete(_request())

    assert error_info.value.error.code == "retryable_http_error"
    assert error_info.value.error.retryable is True
    assert error_info.value.error.attempts == 2


def test_fake_can_raise_configured_model_error() -> None:
    configured = LLMError(code="model_down", message="offline", retryable=True)
    client = FakeLLMClient([configured])

    with pytest.raises(LLMClientError) as error_info:
        client.complete(_request())

    assert error_info.value.error == configured


def test_serializes_assistant_tool_calls_and_tool_observation() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_api_response({"content": "done"}))

    call = ToolCallRequest(
        id="call_1", name="execute_sql", arguments={"sql": "SELECT 1"}
    )
    request = LLMRequest(
        messages=[
            LLMMessage(role="user", content="run"),
            LLMMessage(role="assistant", content=None, tool_calls=[call]),
            LLMMessage(role="tool", tool_call_id="call_1", content='{"success":true}'),
        ],
        tools=[_tool()],
    )
    client = OpenAICompatibleLLMClient(
        _settings(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    client.complete(request)

    messages = bodies[0]["messages"]
    assert messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "arguments": '{"sql":"SELECT 1"}',
                },
            }
        ],
    }
    assert messages[2] == {
        "role": "tool",
        "content": '{"success":true}',
        "tool_call_id": "call_1",
    }


def test_tools_fallback_drops_conflicting_business_response_schema() -> None:
    bodies: list[dict[str, object]] = []
    fallback = json.dumps(
        {
            "type": "tool_calls",
            "tool_calls": [
                {"id": "fallback", "name": "inspect_schema", "arguments": {}}
            ],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(400, text="tools are not supported")
        return httpx.Response(200, json=_api_response({"content": fallback}))

    client = OpenAICompatibleLLMClient(
        _settings(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.complete(
        _request(tools=[_tool()], response_schema={"title": "SQLGeneration"})
    )

    assert response.response_mode is ResponseMode.JSON_FALLBACK
    assert "tools" in bodies[0]
    assert "response_format" in bodies[0]
    assert "tools" not in bodies[1]
    assert "response_format" not in bodies[1]
