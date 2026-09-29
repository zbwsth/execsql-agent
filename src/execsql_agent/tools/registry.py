"""Strict allowlisted tool definitions, argument validation, and dispatch."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from time import perf_counter
from typing import cast

from pydantic import Field, ValidationError

from execsql_agent.models import (
    ExecutionResult,
    StrictModel,
    ToolCallRequest,
    ToolCallResult,
    ToolCallValidation,
    ToolDefinition,
)
from execsql_agent.tools.schema_loader import SchemaLoader
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator


class InspectSchemaArguments(StrictModel):
    """Arguments for optionally filtered schema inspection."""

    table_names: list[str] | None = None


class ListTablesArguments(StrictModel):
    """Arguments for lightweight table discovery."""


class ValidateSQLArguments(StrictModel):
    """Arguments for static and compile-time SQL validation."""

    sql: str = Field(min_length=1)


class ExecuteSQLArguments(StrictModel):
    """Arguments for bounded read-only SQL execution."""

    sql: str = Field(min_length=1)


class ToolRegistry:
    """Expose the validated SQLite tool allowlist."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        validator: SQLValidator | None = None,
        executor: SQLExecutor | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.validator = validator or SQLValidator()
        self.executor = executor or SQLExecutor(
            self.database_path, validator=self.validator
        )
        self._argument_models: dict[str, type[StrictModel]] = {
            "inspect_schema": InspectSchemaArguments,
            "validate_sql": ValidateSQLArguments,
            "execute_sql": ExecuteSQLArguments,
            "list_tables": ListTablesArguments,
        }

    @property
    def definitions(self) -> list[ToolDefinition]:
        """Return the only JSON Schema tools visible to the model."""

        descriptions = {
            "inspect_schema": "Inspect all SQLite tables or selected table names.",
            "validate_sql": "Validate one read-only SQLite SELECT without returning rows.",
            "execute_sql": "Execute one bounded read-only SQLite SELECT query.",
            "list_tables": (
                "List the available tables in the current database. Use this when the "
                "relevant table names are unknown before inspecting detailed schema."
            ),
        }
        return [
            ToolDefinition(
                name=name,
                description=descriptions[name],
                parameters=cast(dict[str, object], model.model_json_schema()),
            )
            for name, model in self._argument_models.items()
        ]

    @staticmethod
    def canonical_signature(call: ToolCallRequest) -> str:
        """Create a stable name plus sorted-JSON signature for repeat checks."""

        if call.arguments is None:
            encoded = f"raw:{(call.raw_arguments or '').strip()}"
        else:
            encoded = json.dumps(
                call.arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=repr,
            )
        return f"{call.name}:{encoded}"

    def validate_call(
        self,
        call: ToolCallRequest,
        *,
        duplicate: bool = False,
        duplicate_code: str = "repeated_tool_call",
    ) -> ToolCallValidation:
        """Validate allowlist membership, JSON arguments, schema, and duplication."""

        signature = self.canonical_signature(call)
        model = self._argument_models.get(call.name)
        if model is None:
            return ToolCallValidation(
                tool_call_id=call.id,
                tool_name=call.name,
                valid=False,
                known_tool=False,
                arguments_valid=call.arguments is not None,
                duplicate=duplicate,
                canonical_signature=signature,
                error_code="unknown_tool",
                error_message=f"Unknown tool: {call.name}",
            )
        if call.arguments is None:
            return ToolCallValidation(
                tool_call_id=call.id,
                tool_name=call.name,
                valid=False,
                known_tool=True,
                arguments_valid=False,
                duplicate=duplicate,
                canonical_signature=signature,
                error_code="invalid_json_arguments",
                error_message="Tool arguments are not a valid JSON object.",
            )
        try:
            model.model_validate(call.arguments)
        except ValidationError as error:
            return ToolCallValidation(
                tool_call_id=call.id,
                tool_name=call.name,
                valid=False,
                known_tool=True,
                arguments_valid=False,
                duplicate=duplicate,
                canonical_signature=signature,
                error_code="invalid_arguments",
                error_message=str(error),
            )
        if duplicate:
            return ToolCallValidation(
                tool_call_id=call.id,
                tool_name=call.name,
                valid=False,
                known_tool=True,
                arguments_valid=True,
                duplicate=True,
                canonical_signature=signature,
                error_code=duplicate_code,
                error_message="The tool call duplicates an earlier call without new state.",
            )
        return ToolCallValidation(
            tool_call_id=call.id,
            tool_name=call.name,
            valid=True,
            known_tool=True,
            arguments_valid=True,
            canonical_signature=signature,
        )

    def dispatch(
        self,
        call: ToolCallRequest,
        *,
        duplicate: bool = False,
        duplicate_code: str = "repeated_tool_call",
    ) -> tuple[ToolCallValidation, ToolCallResult]:
        """Validate then synchronously execute one allowlisted tool."""

        started_at = perf_counter()
        validation = self.validate_call(
            call, duplicate=duplicate, duplicate_code=duplicate_code
        )
        if not validation.valid:
            return validation, ToolCallResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=False,
                executed=False,
                error_code=validation.error_code,
                error_message=validation.error_message,
                duration_ms=(perf_counter() - started_at) * 1000,
            )

        try:
            if call.name == "list_tables":
                result = self._list_tables(
                    ListTablesArguments.model_validate(call.arguments)
                )
            elif call.name == "inspect_schema":
                result = self._inspect_schema(
                    InspectSchemaArguments.model_validate(call.arguments)
                )
            elif call.name == "validate_sql":
                result = self._validate_sql(
                    ValidateSQLArguments.model_validate(call.arguments)
                )
            else:
                result = self._execute_sql(
                    ExecuteSQLArguments.model_validate(call.arguments)
                )
        except (OSError, sqlite3.Error, ValueError) as error:
            result = ToolCallResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=False,
                executed=False,
                error_code="tool_error",
                error_message=str(error),
                duration_ms=0,
            )
        return validation, result.model_copy(
            update={
                "tool_call_id": call.id,
                "tool_name": call.name,
                "duration_ms": (perf_counter() - started_at) * 1000,
            }
        )

    def _list_tables(self, arguments: ListTablesArguments) -> ToolCallResult:
        del arguments
        tables = SchemaLoader(self.database_path).list_tables()
        return ToolCallResult(
            tool_call_id="placeholder",
            tool_name="list_tables",
            success=True,
            executed=True,
            output={"tables": tables, "count": len(tables)},
            duration_ms=0,
        )

    def _inspect_schema(self, arguments: InspectSchemaArguments) -> ToolCallResult:
        schema = SchemaLoader(self.database_path).load()
        selected = schema.tables
        if arguments.table_names:
            requested = set(arguments.table_names)
            selected = [table for table in schema.tables if table.name in requested]
            missing = sorted(requested - {table.name for table in selected})
            if missing:
                return ToolCallResult(
                    tool_call_id="placeholder",
                    tool_name="inspect_schema",
                    success=False,
                    executed=True,
                    output={
                        "database_id": schema.database_id,
                        "tables": [
                            table.model_dump(mode="json") for table in selected
                        ],
                    },
                    error_code="unknown_table",
                    error_message=f"Unknown table(s): {', '.join(missing)}",
                    duration_ms=0,
                )
        output: dict[str, object] = {
            "database_id": schema.database_id,
            "tables": [table.model_dump(mode="json") for table in selected],
        }
        if not arguments.table_names:
            output["schema_text"] = schema.summary_text
        return ToolCallResult(
            tool_call_id="placeholder",
            tool_name="inspect_schema",
            success=True,
            executed=True,
            output=output,
            duration_ms=0,
        )

    def _validate_sql(self, arguments: ValidateSQLArguments) -> ToolCallResult:
        safety = self.validator.validate(
            arguments.sql, database_path=self.database_path
        )
        output: dict[str, object] = {
            "safety_check": safety.model_dump(mode="json")
        }
        if not safety.safe:
            return ToolCallResult(
                tool_call_id="placeholder",
                tool_name="validate_sql",
                success=False,
                executed=False,
                output=output,
                error_code="unsafe_sql",
                error_message=safety.reason,
                duration_ms=0,
            )
        if safety.syntax_valid is False:
            return ToolCallResult(
                tool_call_id="placeholder",
                tool_name="validate_sql",
                success=False,
                executed=True,
                output=output,
                error_code="sql_validation_error",
                error_message=safety.validation_error,
                duration_ms=0,
            )
        return ToolCallResult(
            tool_call_id="placeholder",
            tool_name="validate_sql",
            success=True,
            executed=True,
            output=output,
            duration_ms=0,
        )

    def _execute_sql(self, arguments: ExecuteSQLArguments) -> ToolCallResult:
        execution = self.executor.execute(arguments.sql)
        output: dict[str, object] = {
            "execution_result": execution.model_dump(mode="json")
        }
        if not execution.execution_success:
            error_code = "unsafe_sql" if not execution.executed else "execution_error"
            error_message = (
                execution.blocked_reason
                if not execution.executed
                else execution.error.message if execution.error else "Execution failed."
            )
            return ToolCallResult(
                tool_call_id="placeholder",
                tool_name="execute_sql",
                success=False,
                executed=execution.executed,
                output=output,
                error_code=error_code,
                error_message=error_message,
                duration_ms=0,
            )
        return ToolCallResult(
            tool_call_id="placeholder",
            tool_name="execute_sql",
            success=True,
            executed=True,
            output=output,
            duration_ms=0,
        )

    @staticmethod
    def execution_result(result: ToolCallResult) -> ExecutionResult | None:
        """Recover a typed ExecutionResult from execute_sql output."""

        if result.output is None:
            return None
        raw = result.output.get("execution_result")
        if not isinstance(raw, dict):
            return None
        return ExecutionResult.model_validate(raw)
