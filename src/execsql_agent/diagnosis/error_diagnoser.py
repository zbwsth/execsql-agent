"""Rule-first SQLite error diagnosis with an optional LLM fallback."""

from __future__ import annotations

import json
import re
from difflib import get_close_matches
from typing import cast

from pydantic import ValidationError

from execsql_agent.llm.base import LLMClient
from execsql_agent.models import (
    DatabaseSchema,
    DiagnosisSource,
    ErrorDiagnosis,
    ErrorType,
    LLMCallConfig,
    LLMMessage,
    LLMRequest,
)


class DiagnosisError(ValueError):
    """Raised when an LLM fallback cannot produce a valid diagnosis."""


class ErrorDiagnoser:
    """Diagnose explicit SQLite errors with deterministic rules first."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    def diagnose(
        self,
        *,
        sql: str,
        schema: DatabaseSchema,
        error_message: str,
        forced_error_type: ErrorType | None = None,
    ) -> ErrorDiagnosis:
        """Return a rule diagnosis or use the configured LLM for unknown errors."""

        rule_result = self._diagnose_by_rule(
            sql=sql,
            schema=schema,
            error_message=error_message,
            forced_error_type=forced_error_type,
        )
        if rule_result.error_type is not ErrorType.UNKNOWN_ERROR or self.client is None:
            return rule_result
        return self._diagnose_with_llm(sql, schema, error_message)

    def _diagnose_by_rule(
        self,
        *,
        sql: str,
        schema: DatabaseSchema,
        error_message: str,
        forced_error_type: ErrorType | None,
    ) -> ErrorDiagnosis:
        if forced_error_type is ErrorType.UNSAFE_SQL:
            return ErrorDiagnosis(
                error_type=ErrorType.UNSAFE_SQL,
                cause=error_message,
                repair_instruction="Generate one read-only SELECT query using the supplied schema.",
                source=DiagnosisSource.RULE,
            )
        if forced_error_type is ErrorType.REPEATED_SQL:
            return ErrorDiagnosis(
                error_type=ErrorType.REPEATED_SQL,
                cause="The normalized SQL is identical to an earlier candidate.",
                repair_instruction="Generate a materially different SQL query.",
                source=DiagnosisSource.RULE,
            )
        if forced_error_type is ErrorType.EMPTY_RESULT:
            return ErrorDiagnosis(
                error_type=ErrorType.EMPTY_RESULT,
                cause="The query returned no rows.",
                repair_instruction="Recheck filters and joins without inventing schema fields.",
                source=DiagnosisSource.RULE,
            )

        missing_table = re.search(r"no such table:\s*([^\s]+)", error_message, re.I)
        if missing_table:
            requested = missing_table.group(1).strip("`\"[]")
            table_names = [table.name for table in schema.tables]
            related = get_close_matches(requested, table_names, n=3, cutoff=0.3)
            suggestion = f" Likely tables: {', '.join(related)}." if related else ""
            return ErrorDiagnosis(
                error_type=ErrorType.MISSING_TABLE,
                cause=f"Table {requested!r} does not exist.{suggestion}",
                repair_instruction="Use an existing table name from the schema.",
                related_tables=related,
                source=DiagnosisSource.RULE,
            )

        missing_column = re.search(r"no such column:\s*([^\s]+)", error_message, re.I)
        if missing_column:
            requested = missing_column.group(1).strip("`\"[]")
            short_name = requested.rsplit(".", maxsplit=1)[-1]
            columns = [
                column.name for table in schema.tables for column in table.columns
            ]
            related_columns = get_close_matches(short_name, columns, n=3, cutoff=0.3)
            related_tables = [
                table.name
                for table in schema.tables
                if any(column.name in related_columns for column in table.columns)
            ]
            return ErrorDiagnosis(
                error_type=ErrorType.MISSING_COLUMN,
                cause=f"Column {requested!r} does not exist in the referenced scope.",
                repair_instruction=(
                    "Use an existing column and qualify it with its table if needed."
                ),
                related_tables=related_tables,
                related_columns=related_columns,
                source=DiagnosisSource.RULE,
            )

        ambiguous = re.search(
            r"ambiguous column name:\s*([^\s]+)", error_message, re.I
        )
        if ambiguous:
            column = ambiguous.group(1).strip("`\"[]")
            return ErrorDiagnosis(
                error_type=ErrorType.AMBIGUOUS_COLUMN,
                cause=f"Column {column!r} exists in more than one query source.",
                repair_instruction="Qualify the column with the correct table or alias.",
                related_columns=[column],
                source=DiagnosisSource.RULE,
            )

        if re.search(r"syntax error|incomplete input|unrecognized token", error_message, re.I):
            return ErrorDiagnosis(
                error_type=ErrorType.SYNTAX_ERROR,
                cause=error_message,
                repair_instruction=(
                    "Correct the SQLite syntax while preserving the question intent."
                ),
                source=DiagnosisSource.RULE,
            )

        return ErrorDiagnosis(
            error_type=ErrorType.UNKNOWN_ERROR,
            cause=error_message or "The failure could not be classified by deterministic rules.",
            repair_instruction="Review the SQL and SQLite error before generating a repair.",
            source=DiagnosisSource.RULE,
        )

    def _diagnose_with_llm(
        self, sql: str, schema: DatabaseSchema, error_message: str
    ) -> ErrorDiagnosis:
        if self.client is None:
            raise DiagnosisError("LLM diagnosis requested without an LLM client.")
        payload = {
            "sql": sql,
            "sqlite_error": error_message,
            "schema": schema.summary_text,
            "allowed_error_types": [error_type.value for error_type in ErrorType],
            "warning": (
                "incorrect_join and aggregation_error are hypotheses, not reliable rule labels"
            ),
        }
        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "Diagnose the SQL failure. Return JSON only with error_type, cause, "
                        "repair_instruction, related_tables, related_columns, and source."
                    ),
                ),
                LLMMessage(
                    role="user", content=json.dumps(payload, ensure_ascii=False)
                ),
            ],
            config=LLMCallConfig(
                response_schema_name="error_diagnosis",
                response_schema=cast(
                    dict[str, object], ErrorDiagnosis.model_json_schema()
                ),
            ),
        )
        response = self.client.complete(request)
        if response.tool_calls or response.final_answer is None:
            raise DiagnosisError("LLM diagnosis must be a JSON text response.")
        try:
            parsed = ErrorDiagnosis.model_validate_json(response.final_answer)
        except ValidationError as error:
            raise DiagnosisError(f"Invalid ErrorDiagnosis response: {error}") from error
        return parsed.model_copy(update={"source": DiagnosisSource.LLM})
