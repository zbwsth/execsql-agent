"""Bounded, read-only SQLite query execution."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from time import perf_counter
from typing import cast

from execsql_agent.models import ExecutionError, ExecutionResult
from execsql_agent.tools.sql_validator import SQLValidator

_DENIED_AUTHORIZER_ACTIONS = {
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ANALYZE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_CREATE_VTABLE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_DROP_VTABLE,
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_PRAGMA,
    sqlite3.SQLITE_REINDEX,
    sqlite3.SQLITE_SAVEPOINT,
    sqlite3.SQLITE_TRANSACTION,
    sqlite3.SQLITE_UPDATE,
}


def _readonly_uri(database_path: Path) -> str:
    return f"{database_path.resolve().as_uri()}?mode=ro"


def _read_only_authorizer(
    action_code: int,
    _argument_1: str | None,
    _argument_2: str | None,
    _database_name: str | None,
    _trigger_name: str | None,
) -> int:
    if action_code in _DENIED_AUTHORIZER_ACTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _sqlite_error(error: sqlite3.Error) -> ExecutionError:
    raw_name = getattr(error, "sqlite_errorname", None)
    raw_code = getattr(error, "sqlite_errorcode", None)
    return ExecutionError(
        message=str(error),
        sqlite_error_name=raw_name if isinstance(raw_name, str) else None,
        sqlite_error_code=raw_code if isinstance(raw_code, int) else None,
    )


class SQLExecutor:
    """Execute one validated SELECT against a SQLite database in read-only mode."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        max_rows: int = 100,
        progress_handler_ops: int = 10_000,
        progress_handler_max_callbacks: int = 1_000,
        validator: SQLValidator | None = None,
    ) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        if progress_handler_ops < 1 or progress_handler_max_callbacks < 1:
            raise ValueError("progress handler limits must be at least 1")
        self.database_path = Path(database_path)
        self.max_rows = max_rows
        self.progress_handler_ops = progress_handler_ops
        self.progress_handler_max_callbacks = progress_handler_max_callbacks
        self.validator = validator or SQLValidator()

    def execute(self, sql: str) -> ExecutionResult:
        """Safely execute SQL and return at most ``max_rows`` rows."""

        return self._execute(sql, max_rows=self.max_rows)

    def execute_full(
        self,
        sql: str,
        *,
        timeout_seconds: float = 30.0,
    ) -> ExecutionResult:
        """Safely execute SQL and return the complete result for evaluation scoring."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return self._execute(
            sql,
            max_rows=None,
            timeout_seconds=timeout_seconds,
        )

    def _execute(
        self,
        sql: str,
        *,
        max_rows: int | None,
        timeout_seconds: float | None = None,
    ) -> ExecutionResult:
        """Execute with shared read-only and resource protections."""

        started_at = perf_counter()
        safety = self.validator.validate(sql)
        if not safety.safe:
            return ExecutionResult(
                executed=False,
                execution_success=False,
                blocked_reason=safety.reason,
                duration_ms=(perf_counter() - started_at) * 1000,
            )

        connection: sqlite3.Connection | None = None
        executed = False
        timed_out = False
        try:
            connection = sqlite3.connect(_readonly_uri(self.database_path), uri=True)
            connection.execute("PRAGMA query_only = ON")
            connection.set_authorizer(_read_only_authorizer)

            callback_count = 0
            def progress_handler() -> int:
                nonlocal callback_count, timed_out
                if timeout_seconds is not None:
                    timed_out = perf_counter() - started_at >= timeout_seconds
                    return int(timed_out)
                callback_count += 1
                return int(callback_count >= self.progress_handler_max_callbacks)

            connection.set_progress_handler(progress_handler, self.progress_handler_ops)
            executed = True
            cursor = connection.execute(sql)
            if max_rows is None:
                raw_rows = cast(list[tuple[object, ...]], cursor.fetchall())
                truncated = False
                rows = [list(row) for row in raw_rows]
            else:
                raw_rows = cast(
                    list[tuple[object, ...]], cursor.fetchmany(max_rows + 1)
                )
                truncated = len(raw_rows) > max_rows
                rows = [list(row) for row in raw_rows[:max_rows]]
            columns = [str(description[0]) for description in (cursor.description or [])]
            return ExecutionResult(
                executed=True,
                execution_success=True,
                columns=columns,
                rows=rows,
                returned_row_count=len(rows),
                truncated=truncated,
                duration_ms=(perf_counter() - started_at) * 1000,
            )
        except sqlite3.Error as error:
            execution_error = _sqlite_error(error)
            if timeout_seconds is not None and timed_out:
                execution_error = execution_error.model_copy(
                    update={
                        "message": (
                            f"query timed out after {timeout_seconds:.1f} seconds"
                        )
                    }
                )
            return ExecutionResult(
                executed=executed,
                execution_success=False,
                error=execution_error,
                duration_ms=(perf_counter() - started_at) * 1000,
            )
        finally:
            if connection is not None:
                connection.close()
