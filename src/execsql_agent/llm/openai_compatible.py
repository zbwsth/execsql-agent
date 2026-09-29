"""OpenAI-compatible chat completions client implemented with httpx."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from typing import Self, cast

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from execsql_agent.llm.base import LLMClient, LLMClientError
from execsql_agent.models import (
    LLMError,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
)

_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_CAPABILITY_STATUS_CODES = {400, 404, 422}


class OpenAICompatibleSettings(BaseSettings):
    """Environment-backed connection settings."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_", extra="ignore")

    api_key: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    enable_thinking: bool | None = None


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _as_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object")
    return cast(dict[str, object], value)


def _as_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return cast(list[object], value)


def _strip_json_fence(content: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, re.I | re.S)
    return match.group(1) if match else content.strip()


class OpenAICompatibleLLMClient(LLMClient):
    """Normalize OpenAI-compatible native and fallback responses."""

    def __init__(
        self,
        settings: OpenAICompatibleSettings | None = None,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.0,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0 or retry_backoff_seconds < 0:
            raise ValueError("retry settings must be non-negative")
        if temperature is not None and temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.settings = settings or OpenAICompatibleSettings(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", ""),
            model=os.environ.get("OPENAI_MODEL", ""),
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.endpoint = _chat_completions_url(self.settings.base_url)
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Close an internally created HTTP client."""

        if self._owns_client:
            self._client.close()

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Call the API with capability fallback and bounded retries."""

        use_native_tools = bool(request.tools)
        use_response_format = request.config.response_schema is not None
        json_tool_fallback = False
        transient_failures = 0
        attempts = 0

        while True:
            attempts += 1
            body = self._build_body(
                request,
                use_native_tools=use_native_tools,
                use_response_format=use_response_format,
                json_tool_fallback=json_tool_fallback,
            )
            try:
                response = self._client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.settings.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=self.timeout_seconds,
                )
            except httpx.TimeoutException as error:
                if transient_failures < self.max_retries:
                    transient_failures += 1
                    self._sleep_before_retry()
                    continue
                raise self._client_error(
                    code="timeout",
                    message=f"LLM request timed out: {error}",
                    retryable=True,
                    attempts=attempts,
                ) from error
            except httpx.RequestError as error:
                if transient_failures < self.max_retries:
                    transient_failures += 1
                    self._sleep_before_retry()
                    continue
                raise self._client_error(
                    code="network_error",
                    message=f"LLM request failed: {error}",
                    retryable=True,
                    attempts=attempts,
                ) from error

            if use_response_format and self._feature_rejected(response, "response_format"):
                use_response_format = False
                continue
            if use_native_tools and self._feature_rejected(response, "tools"):
                use_native_tools = False
                use_response_format = False
                json_tool_fallback = True
                continue
            if response.status_code in _RETRYABLE_STATUS_CODES:
                if transient_failures < self.max_retries:
                    transient_failures += 1
                    self._sleep_before_retry()
                    continue
                raise self._client_error(
                    code="retryable_http_error",
                    message=f"LLM service returned HTTP {response.status_code}: {response.text}",
                    retryable=True,
                    attempts=attempts,
                    status_code=response.status_code,
                )
            if response.status_code >= 400:
                raise self._client_error(
                    code="http_error",
                    message=f"LLM service returned HTTP {response.status_code}: {response.text}",
                    retryable=False,
                    attempts=attempts,
                    status_code=response.status_code,
                )

            return self._parse_response(
                response,
                attempts=attempts,
                json_tool_fallback=json_tool_fallback,
            )

    def _build_body(
        self,
        request: LLMRequest,
        *,
        use_native_tools: bool,
        use_response_format: bool,
        json_tool_fallback: bool,
    ) -> dict[str, object]:
        messages: list[dict[str, object]] = []
        if json_tool_fallback:
            definitions = [tool.model_dump(mode="json") for tool in request.tools]
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Native function calling is unavailable. Return JSON only. "
                        "Use either {\"type\":\"tool_calls\",\"tool_calls\":["
                        "{\"id\":\"...\",\"name\":\"...\",\"arguments\":{}}]} "
                        "or {\"type\":\"final\",\"final_answer\":\"...\"}. "
                        f"Available tools: {json.dumps(definitions, ensure_ascii=False)}"
                    ),
                }
            )
        if json_tool_fallback:
            messages.extend(
                self._serialize_fallback_message(message) for message in request.messages
            )
        else:
            messages.extend(self._serialize_message(message) for message in request.messages)

        body: dict[str, object] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": (
                request.config.temperature
                if self.temperature is None
                else self.temperature
            ),
            "max_tokens": (
                request.config.max_tokens
                if self.max_tokens is None
                else self.max_tokens
            ),
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.settings.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.settings.enable_thinking}
        if use_native_tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
        if (
            use_response_format
            and not json_tool_fallback
            and request.config.response_schema is not None
        ):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.config.response_schema_name,
                    "schema": request.config.response_schema,
                    "strict": True,
                },
            }
        return body

    @staticmethod
    def _serialize_message(message: LLMMessage) -> dict[str, object]:
        body: dict[str, object] = {"role": message.role, "content": message.content}
        if message.role == "assistant" and message.tool_calls:
            body["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments
                        if call.raw_arguments is not None
                        else json.dumps(
                            call.arguments or {},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        if message.role == "tool":
            body["tool_call_id"] = message.tool_call_id
        return body

    @staticmethod
    def _serialize_fallback_message(message: LLMMessage) -> dict[str, object]:
        if message.role == "assistant" and message.tool_calls:
            calls = [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                    if call.arguments is not None
                    else call.raw_arguments,
                }
                for call in message.tool_calls
            ]
            return {
                "role": "assistant",
                "content": json.dumps(
                    {"type": "tool_calls", "tool_calls": calls},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        if message.role == "tool":
            return {
                "role": "user",
                "content": (
                    f"Tool observation for {message.tool_call_id}: {message.content}"
                ),
            }
        return {"role": message.role, "content": message.content}

    def _parse_response(
        self,
        response: httpx.Response,
        *,
        attempts: int,
        json_tool_fallback: bool,
    ) -> LLMResponse:
        try:
            payload = _as_mapping(cast(object, response.json()), "response")
            choices = _as_list(payload.get("choices"), "response.choices")
            if not choices:
                raise ValueError("response.choices must not be empty")
            choice = _as_mapping(choices[0], "response.choices[0]")
            message = _as_mapping(choice.get("message"), "response message")
            finish_reason_value = choice.get("finish_reason")
            finish_reason = (
                finish_reason_value if isinstance(finish_reason_value, str) else None
            )
            content_value = message.get("content")
            content = content_value if isinstance(content_value, str) else None

            if json_tool_fallback:
                if content is None:
                    raise ValueError("JSON fallback response requires text content")
                return self._parse_json_fallback(content, finish_reason)

            native_calls_value = message.get("tool_calls")
            if native_calls_value is not None:
                native_calls = _as_list(native_calls_value, "message.tool_calls")
                tool_calls = [self._parse_native_tool_call(item) for item in native_calls]
                return LLMResponse(
                    final_answer=content or None,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    response_mode=ResponseMode.NATIVE_TOOL_CALLS,
                    raw_content=content,
                )
            if content:
                return LLMResponse(
                    final_answer=content,
                    finish_reason=finish_reason,
                    response_mode=ResponseMode.PLAIN_FINAL,
                    raw_content=content,
                )
            raise ValueError("response message contains no content or tool_calls")
        except (ValueError, json.JSONDecodeError) as error:
            raise self._client_error(
                code="invalid_response",
                message=f"Invalid LLM response: {error}",
                retryable=False,
                attempts=attempts,
            ) from error

    def _parse_native_tool_call(self, value: object) -> ToolCallRequest:
        call = _as_mapping(value, "tool call")
        call_id = call.get("id")
        function = _as_mapping(call.get("function"), "tool call function")
        name = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool call id is required")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call function name is required")
        if not isinstance(raw_arguments, str):
            raise ValueError("tool call arguments must be a JSON string")
        arguments = self._parse_arguments(raw_arguments)
        return ToolCallRequest(
            id=call_id,
            name=name,
            arguments=arguments,
            raw_arguments=raw_arguments,
        )

    def _parse_json_fallback(
        self, content: str, finish_reason: str | None
    ) -> LLMResponse:
        payload = _as_mapping(
            cast(object, json.loads(_strip_json_fence(content))), "JSON fallback"
        )
        response_type = payload.get("type")
        if response_type == "final":
            final_answer = payload.get("final_answer")
            if not isinstance(final_answer, str) or not final_answer:
                raise ValueError("JSON fallback final_answer is required")
            return LLMResponse(
                final_answer=final_answer,
                finish_reason=finish_reason,
                response_mode=ResponseMode.JSON_FALLBACK,
                raw_content=content,
            )
        if response_type == "tool_calls":
            raw_calls = _as_list(payload.get("tool_calls"), "JSON fallback tool_calls")
            calls: list[ToolCallRequest] = []
            for raw_call in raw_calls:
                call = _as_mapping(raw_call, "JSON fallback tool call")
                call_id = call.get("id")
                name = call.get("name")
                arguments_value = call.get("arguments")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("JSON fallback tool call id is required")
                if not isinstance(name, str) or not name:
                    raise ValueError("JSON fallback tool name is required")
                arguments: dict[str, object] | None
                if isinstance(arguments_value, Mapping):
                    arguments = _as_mapping(
                        arguments_value, "JSON fallback arguments"
                    )
                    raw_arguments = None
                elif isinstance(arguments_value, str):
                    arguments = self._parse_arguments(arguments_value)
                    raw_arguments = arguments_value
                else:
                    arguments = None
                    raw_arguments = None
                calls.append(
                    ToolCallRequest(
                        id=call_id,
                        name=name,
                        arguments=arguments,
                        raw_arguments=raw_arguments,
                    )
                )
            if not calls:
                raise ValueError("JSON fallback tool_calls must not be empty")
            return LLMResponse(
                tool_calls=calls,
                finish_reason=finish_reason,
                response_mode=ResponseMode.JSON_FALLBACK,
                raw_content=content,
            )
        raise ValueError("JSON fallback type must be 'tool_calls' or 'final'")

    @staticmethod
    def _parse_arguments(raw_arguments: str) -> dict[str, object] | None:
        try:
            parsed = cast(object, json.loads(raw_arguments))
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, Mapping):
            return None
        return _as_mapping(parsed, "tool call arguments")

    @staticmethod
    def _feature_rejected(response: httpx.Response, feature: str) -> bool:
        if response.status_code not in _CAPABILITY_STATUS_CODES:
            return False
        body = response.text.lower()
        rejection_words = ("unsupported", "not supported", "unknown", "unrecognized")
        return feature.lower() in body and any(word in body for word in rejection_words)

    def _sleep_before_retry(self) -> None:
        if self.retry_backoff_seconds > 0:
            time.sleep(self.retry_backoff_seconds)

    @staticmethod
    def _client_error(
        *,
        code: str,
        message: str,
        retryable: bool,
        attempts: int,
        status_code: int | None = None,
    ) -> LLMClientError:
        return LLMClientError(
            LLMError(
                code=code,
                message=message,
                retryable=retryable,
                attempts=attempts,
                status_code=status_code,
            )
        )
