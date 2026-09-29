"""Strict SQL extraction for raw GRPO completions and compatible tool calls."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import cast

from execsql_agent.rlvr.models import (
    ParseSource,
    ParseStatus,
    ResponseParseResult,
    VerifierFailureKind,
)
from execsql_agent.tools.sql_validator import SQLValidator

_PLAIN_SQL_START = re.compile(r"^(?:SELECT|WITH)\b", re.IGNORECASE)
_QWEN_TOOL_CALL = re.compile(r"^\s*<tool_call>\s*(.*?)\s*</tool_call>\s*$", re.DOTALL)
_QWEN_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


def _failure(kind: VerifierFailureKind, detail: str) -> ResponseParseResult:
    return ResponseParseResult(
        parse_status=ParseStatus.FAILED,
        failure_kind=kind,
        detail=detail,
    )


def _contains_database_path(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() == "database_path" or _contains_database_path(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_database_path(item) for item in value)
    return False


def _single_statement(sql: str, source: ParseSource) -> ResponseParseResult:
    candidate = sql.strip()
    if not candidate:
        return _failure(VerifierFailureKind.MALFORMED_RESPONSE, "SQL must not be empty.")
    scan = SQLValidator().validate(candidate)
    if scan.blocked_operation == "MULTI_STATEMENT":
        return _failure(
            VerifierFailureKind.MULTIPLE_SQL_CANDIDATES,
            "The completion contains more than one SQL statement.",
        )
    return ResponseParseResult(
        parse_status=ParseStatus.PARSED,
        sql=candidate,
        source=source,
    )


def _arguments_object(value: object) -> Mapping[str, object] | None:
    if isinstance(value, str):
        try:
            value = cast(object, json.loads(value))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, Mapping) else None


def _parse_tool_call(call_value: object) -> ResponseParseResult:
    if not isinstance(call_value, Mapping):
        return _failure(VerifierFailureKind.MALFORMED_RESPONSE, "Tool call must be an object.")
    if _contains_database_path(call_value):
        return _failure(
            VerifierFailureKind.RESPONSE_DATABASE_PATH,
            "A model response may not provide database_path.",
        )

    if "function" in call_value:
        allowed = {"id", "type", "function"}
        if set(call_value) - allowed:
            return _failure(
                VerifierFailureKind.MALFORMED_RESPONSE,
                "Tool call contains unsupported fields.",
            )
        function = call_value.get("function")
        if not isinstance(function, Mapping) or set(function) != {"name", "arguments"}:
            return _failure(
                VerifierFailureKind.MALFORMED_RESPONSE,
                "Function call must contain exactly name and arguments.",
            )
        name = function.get("name")
        arguments_value = function.get("arguments")
    else:
        allowed = {"id", "type", "name", "arguments"}
        if set(call_value) - allowed or not {"name", "arguments"} <= set(call_value):
            return _failure(
                VerifierFailureKind.MALFORMED_RESPONSE,
                "Tool call must contain name and arguments.",
            )
        name = call_value.get("name")
        arguments_value = call_value.get("arguments")

    if name != "execute_sql":
        return _failure(
            VerifierFailureKind.NON_EXECUTE_SQL_TOOL,
            "Only an execute_sql tool call can be verified.",
        )
    arguments = _arguments_object(arguments_value)
    if arguments is None or set(arguments) != {"sql"}:
        return _failure(
            VerifierFailureKind.MALFORMED_RESPONSE,
            "execute_sql arguments must contain exactly one sql field.",
        )
    sql = arguments.get("sql")
    if not isinstance(sql, str):
        return _failure(
            VerifierFailureKind.MALFORMED_RESPONSE,
            "execute_sql sql must be a string.",
        )
    return _single_statement(sql, ParseSource.EXECUTE_SQL_TOOL_CALL)


def _parse_mapping(payload: Mapping[str, object]) -> ResponseParseResult:
    if _contains_database_path(payload):
        return _failure(
            VerifierFailureKind.RESPONSE_DATABASE_PATH,
            "A model response may not provide database_path.",
        )
    if set(payload) == {"sql"}:
        sql = payload.get("sql")
        if not isinstance(sql, str):
            return _failure(
                VerifierFailureKind.MALFORMED_RESPONSE,
                "The canonical sql field must be a string.",
            )
        return _single_statement(sql, ParseSource.JSON)

    if "tool_calls" in payload:
        allowed = {"role", "content", "tool_calls"}
        if set(payload) - allowed:
            return _failure(
                VerifierFailureKind.MALFORMED_RESPONSE,
                "Tool-call envelope contains unsupported fields.",
            )
        calls = payload.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            kind = (
                VerifierFailureKind.MULTIPLE_SQL_CANDIDATES
                if isinstance(calls, list) and len(calls) > 1
                else VerifierFailureKind.MALFORMED_RESPONSE
            )
            return _failure(kind, "Tool-call envelope must contain exactly one call.")
        return _parse_tool_call(calls[0])

    if "name" in payload or "function" in payload:
        return _parse_tool_call(payload)
    return _failure(
        VerifierFailureKind.MALFORMED_RESPONSE,
        "JSON response must be canonical SQL or one reliable tool call.",
    )


def parse_sql_response(completion: str | Mapping[str, object]) -> ResponseParseResult:
    """Extract exactly one SQL candidate without guessing from surrounding prose."""

    if isinstance(completion, Mapping):
        return _parse_mapping(completion)
    if not isinstance(completion, str):
        return _failure(
            VerifierFailureKind.MALFORMED_RESPONSE,
            "Completion must be text or a mapping.",
        )

    stripped = completion.strip()
    if not stripped:
        return _failure(VerifierFailureKind.MALFORMED_RESPONSE, "Completion is empty.")

    qwen_match = _QWEN_TOOL_CALL.fullmatch(stripped)
    if qwen_match:
        try:
            payload = cast(object, json.loads(qwen_match.group(1)))
        except json.JSONDecodeError as error:
            return _failure(
                VerifierFailureKind.MALFORMED_RESPONSE,
                f"Invalid tool-call JSON: {error.msg}.",
            )
        return _parse_tool_call(payload)

    if stripped.startswith(("{", "[")):
        try:
            payload = cast(object, json.loads(stripped))
        except json.JSONDecodeError as error:
            return _failure(
                VerifierFailureKind.MALFORMED_RESPONSE,
                f"Invalid response JSON: {error.msg}.",
            )
        if not isinstance(payload, Mapping):
            return _failure(
                VerifierFailureKind.MALFORMED_RESPONSE,
                "Response JSON must be an object.",
            )
        return _parse_mapping(payload)

    if _PLAIN_SQL_START.match(stripped):
        return _single_statement(stripped, ParseSource.PLAIN_SQL)
    return _failure(
        VerifierFailureKind.MALFORMED_RESPONSE,
        "Completion is not canonical JSON, an execute_sql call, or plain SQL.",
    )


def parse_agentic_sql_response(
    completion: str | Mapping[str, object],
) -> ResponseParseResult:
    """Extract the last valid execute_sql call from a Qwen multi-turn transcript.

    VERL's ToolAgentLoop decodes the complete response span for reward scoring:
    assistant tool calls, zero-loss tool observations, and the final assistant
    answer. The final database-backed SQL is therefore the last execute_sql
    tool call, matching the Harness-v2 final-SQL convention. This parser never
    guesses SQL from observations or prose.
    """

    if isinstance(completion, Mapping):
        return parse_sql_response(completion)
    if not isinstance(completion, str):
        return _failure(
            VerifierFailureKind.MALFORMED_RESPONSE,
            "Agentic completion must be text or a mapping.",
        )

    execute_candidates: list[ResponseParseResult] = []
    for match in _QWEN_TOOL_CALL_BLOCK.finditer(completion):
        try:
            payload = cast(object, json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
        parsed = _parse_tool_call(payload)
        if parsed.parse_status is ParseStatus.PARSED:
            execute_candidates.append(parsed)
    if execute_candidates:
        return execute_candidates[-1]
    return _failure(
        VerifierFailureKind.MALFORMED_RESPONSE,
        "Agentic completion contains no valid execute_sql tool call.",
    )
