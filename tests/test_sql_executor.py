"""Tests for bounded, read-only execution against real SQLite."""

import sqlite3
from pathlib import Path

from execsql_agent.tools.sql_executor import SQLExecutor


def test_executes_select_and_returns_columns(demo_db: Path) -> None:
    result = SQLExecutor(demo_db).execute(
        "SELECT customer_id, customer_name FROM customers ORDER BY customer_id LIMIT 2"
    )

    assert result.executed is True
    assert result.execution_success is True
    assert result.columns == ["customer_id", "customer_name"]
    assert result.rows == [[1, "张伟"], [2, "王芳"]]
    assert result.returned_row_count == 2
    assert result.truncated is False
    assert result.error is None
    assert result.duration_ms >= 0


def test_truncation_reports_only_returned_rows(demo_db: Path) -> None:
    result = SQLExecutor(demo_db, max_rows=3).execute(
        "SELECT customer_id FROM customers ORDER BY customer_id"
    )

    assert result.execution_success is True
    assert result.rows == [[1], [2], [3]]
    assert result.returned_row_count == 3
    assert result.truncated is True


def test_empty_result_is_execution_success(demo_db: Path) -> None:
    result = SQLExecutor(demo_db).execute(
        "SELECT customer_id FROM customers WHERE customer_id < 0"
    )

    assert result.executed is True
    assert result.execution_success is True
    assert result.rows == []
    assert result.returned_row_count == 0
    assert result.truncated is False


def test_captures_real_sqlite_execution_error(demo_db: Path) -> None:
    result = SQLExecutor(demo_db).execute("SELECT * FROM customer")

    assert result.executed is True
    assert result.execution_success is False
    assert result.error is not None
    assert result.error.error_type == "execution_error"
    assert "no such table" in result.error.message


def test_blocks_unsafe_sql_without_execution(demo_db: Path) -> None:
    result = SQLExecutor(demo_db).execute("DELETE FROM customers")

    assert result.executed is False
    assert result.execution_success is False
    assert result.error is None
    assert "not allowed" in (result.blocked_reason or "")

    with sqlite3.connect(demo_db) as connection:
        customer_count = connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    assert customer_count == 10


def test_progress_handler_interrupts_obviously_long_query(demo_db: Path) -> None:
    executor = SQLExecutor(
        demo_db,
        progress_handler_ops=1,
        progress_handler_max_callbacks=1,
    )
    result = executor.execute(
        """
        WITH RECURSIVE counter(value) AS (
            VALUES(0)
            UNION ALL
            SELECT value + 1 FROM counter WHERE value < 1000000
        )
        SELECT SUM(value) FROM counter
        """
    )

    assert result.executed is True
    assert result.execution_success is False
    assert result.error is not None
    assert "interrupt" in result.error.message.lower()


def test_execute_full_returns_all_rows_while_execute_stays_bounded(
    tmp_path: Path,
) -> None:
    database = tmp_path / "many_rows.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE numbers (value INTEGER NOT NULL)")
        connection.executemany(
            "INSERT INTO numbers VALUES (?)",
            [(value,) for value in range(200)],
        )

    executor = SQLExecutor(database)
    bounded = executor.execute("SELECT value FROM numbers ORDER BY value")
    full = executor.execute_full("SELECT value FROM numbers ORDER BY value")

    assert bounded.execution_success is True
    assert bounded.returned_row_count == 100
    assert bounded.truncated is True
    assert full.execution_success is True
    assert full.returned_row_count == 200
    assert full.rows[-1] == [199]
    assert full.truncated is False


def test_execute_full_ignores_callback_count_limit(demo_db: Path) -> None:
    executor = SQLExecutor(
        demo_db,
        progress_handler_ops=1,
        progress_handler_max_callbacks=1,
    )
    result = executor.execute_full(
        """
        WITH RECURSIVE counter(value) AS (
            VALUES(0)
            UNION ALL
            SELECT value + 1 FROM counter WHERE value < 5000
        )
        SELECT SUM(value) FROM counter
        """,
        timeout_seconds=5.0,
    )

    assert result.execution_success is True
    assert result.rows == [[12_502_500]]


def test_execute_full_reports_wall_clock_timeout(demo_db: Path) -> None:
    executor = SQLExecutor(
        demo_db,
        progress_handler_ops=1,
        progress_handler_max_callbacks=1_000_000,
    )
    result = executor.execute_full(
        """
        WITH RECURSIVE counter(value) AS (
            VALUES(0)
            UNION ALL
            SELECT value + 1 FROM counter WHERE value < 1000000
        )
        SELECT SUM(value) FROM counter
        """,
        timeout_seconds=0.000001,
    )

    assert result.executed is True
    assert result.execution_success is False
    assert result.error is not None
    assert result.error.message == "query timed out after 0.0 seconds"


def test_execute_full_allows_normal_aggregate(demo_db: Path) -> None:
    result = SQLExecutor(
        demo_db,
        progress_handler_ops=1,
        progress_handler_max_callbacks=1,
    ).execute_full("SELECT COUNT(*) FROM customers", timeout_seconds=1.0)

    assert result.execution_success is True
    assert result.rows == [[10]]


def test_execute_full_requires_positive_timeout(demo_db: Path) -> None:
    executor = SQLExecutor(demo_db)

    try:
        executor.execute_full("SELECT 1", timeout_seconds=0)
    except ValueError as error:
        assert str(error) == "timeout_seconds must be positive"
    else:
        raise AssertionError("execute_full accepted a non-positive timeout")


def test_missing_database_returns_structured_error(tmp_path: Path) -> None:
    result = SQLExecutor(tmp_path / "missing.db").execute("SELECT 1")

    assert result.executed is False
    assert result.execution_success is False
    assert result.error is not None
