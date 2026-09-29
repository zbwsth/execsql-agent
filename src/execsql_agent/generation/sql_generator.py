"""Prompt construction and validated parsing for SQL generation."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import cast

from pydantic import ValidationError

from execsql_agent.llm.base import LLMClient
from execsql_agent.models import (
    DatabaseSchema,
    ErrorDiagnosis,
    ExecutionError,
    GenerationMode,
    LLMCallConfig,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    SQLGeneration,
)


class SQLGenerationError(ValueError):
    """Raised when model text cannot be validated as SQLGeneration."""


def _strip_json_fence(content: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, re.I | re.S)
    return match.group(1) if match else content.strip()


class SQLGenerator:
    """Generate initial or repaired SQL through an abstract LLMClient."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def generate(
        self,
        *,
        question: str,
        schema: DatabaseSchema,
        mode: GenerationMode,
        history_sql: Sequence[str] = (),
        previous_error: ExecutionError | str | None = None,
        diagnosis: ErrorDiagnosis | None = None,
    ) -> SQLGeneration:
        """Generate and validate one SQL candidate."""

        request = self.build_request(
            question=question,
            schema=schema,
            mode=mode,
            history_sql=history_sql,
            previous_error=previous_error,
            diagnosis=diagnosis,
        )
        return self.parse_response(self.client.complete(request))

    def build_request(
        self,
        *,
        question: str,
        schema: DatabaseSchema,
        mode: GenerationMode,
        history_sql: Sequence[str],
        previous_error: ExecutionError | str | None,
        diagnosis: ErrorDiagnosis | None,
    ) -> LLMRequest:
        """Build the provider-neutral generation prompt without parsing output."""

        error_payload: object
        if isinstance(previous_error, ExecutionError):
            error_payload = previous_error.model_dump(mode="json")
        else:
            error_payload = previous_error
        payload = {
            "mode": mode.value,
            "question": question,
            "schema": schema.summary_text,
            "history_sql": list(history_sql),
            "previous_error": error_payload,
            "diagnosis": diagnosis.model_dump(mode="json") if diagnosis else None,
        }
        system_prompt = (
            "You generate exactly one safe, read-only SQLite SELECT query. "
            "Use only tables and columns present in the supplied schema. "
            "Return JSON only with fields sql, reason, referenced_tables, "
            "and referenced_columns. Do not use Markdown."
        )
        if mode is GenerationMode.REPAIR:
            system_prompt += " Repair the previous SQL according to the diagnosis."
        return LLMRequest(
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user", content=json.dumps(payload, ensure_ascii=False)
                ),
            ],
            config=LLMCallConfig(
                response_schema_name="sql_generation",
                response_schema=cast(
                    dict[str, object], SQLGeneration.model_json_schema()
                ),
            ),
        )

    @staticmethod
    def parse_response(response: LLMResponse) -> SQLGeneration:
        """Parse model text separately from prompt construction."""

        if response.tool_calls:
            raise SQLGenerationError("SQLGenerator does not accept tool_calls responses.")
        if response.final_answer is None:
            raise SQLGenerationError("SQLGenerator response has no structured content.")
        try:
            payload = json.loads(_strip_json_fence(response.final_answer))
            return SQLGeneration.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as error:
            raise SQLGenerationError(f"Invalid SQLGeneration response: {error}") from error
