"""Tests for SQL normalization, safety, and compile validation."""

from pathlib import Path

import pytest

from execsql_agent.tools.sql_validator import SQLValidator


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO customers VALUES (11, 'A', 'B', '2025-01-01')",
        "UPDATE customers SET city = '北京'",
        "DELETE FROM customers",
        "DROP TABLE customers",
        "ALTER TABLE customers ADD COLUMN note TEXT",
        "CREATE TABLE secrets(value TEXT)",
        "REPLACE INTO customers VALUES (1, 'A', 'B', '2025-01-01')",
        "ATTACH DATABASE 'other.db' AS other",
        "DETACH DATABASE other",
        "VACUUM",
        "PRAGMA user_version",
        "WITH doomed AS (SELECT 1) DELETE FROM customers",
    ],
)
def test_blocks_unsafe_operations(sql: str) -> None:
    result = SQLValidator().validate(sql)

    assert result.safe is False
    assert result.blocked_operation is not None


def test_allows_select_and_cte() -> None:
    validator = SQLValidator()

    assert validator.validate("SELECT * FROM customers").safe is True
    assert validator.validate("WITH ids AS (SELECT 1 AS id) SELECT id FROM ids").safe is True


def test_ignores_keywords_inside_strings_and_comments() -> None:
    sql = "SELECT 'DELETE FROM customers' AS note -- DROP TABLE products\n"
    result = SQLValidator().validate(sql)

    assert result.safe is True
    assert "drop table" not in result.normalized_sql


def test_normalization_preserves_string_literal_case() -> None:
    validator = SQLValidator()

    upper = validator.validate("SELECT 'A'").normalized_sql
    lower = validator.validate("SELECT 'a'").normalized_sql

    assert upper != lower
    assert "'A'" in upper
    assert "'a'" in lower


def test_normalization_unifies_unquoted_case_and_whitespace() -> None:
    validator = SQLValidator()

    first = validator.validate(" SELECT  Customer_ID FROM customers ").normalized_sql
    second = validator.validate("select customer_id   from   CUSTOMERS;").normalized_sql

    assert first == second


@pytest.mark.parametrize(
    "quoted_value",
    ["'-- not a comment'", "'/* not a comment */'", "'one;two'"],
)
def test_comment_and_statement_markers_inside_strings_are_preserved(
    quoted_value: str,
) -> None:
    result = SQLValidator().validate(f"SELECT {quoted_value}")

    assert result.safe is True
    assert quoted_value in result.normalized_sql


def test_comments_do_not_affect_normalized_sql() -> None:
    validator = SQLValidator()

    plain = validator.validate("SELECT customer_id FROM customers").normalized_sql
    commented = validator.validate(
        "SELECT /* selected identifier */ customer_id -- source follows\nFROM customers;"
    ).normalized_sql

    assert commented == plain


def test_normalization_preserves_quoted_identifiers() -> None:
    validator = SQLValidator()

    result = validator.validate('SELECT "MixedCase", `BackTick`, [BracketCase] FROM customers')

    assert '"MixedCase"' in result.normalized_sql
    assert "`BackTick`" in result.normalized_sql
    assert "[BracketCase]" in result.normalized_sql


def test_blocks_multiple_statements() -> None:
    result = SQLValidator().validate("SELECT 1; SELECT 2;")

    assert result.safe is False
    assert result.blocked_operation == "MULTI_STATEMENT"


def test_blocks_empty_and_malformed_sql() -> None:
    validator = SQLValidator()

    assert validator.validate("  -- comment only").blocked_operation == "EMPTY_SQL"
    assert validator.validate("SELECT 'unfinished").blocked_operation == "MALFORMED_SQL"


def test_compile_validation_uses_real_database(demo_db: Path) -> None:
    validator = SQLValidator()

    valid = validator.validate(
        "SELECT customer_name FROM customers", database_path=demo_db
    )
    missing_table = validator.validate(
        "SELECT * FROM customer", database_path=demo_db
    )

    assert valid.safe is True
    assert valid.syntax_valid is True
    assert missing_table.safe is True
    assert missing_table.syntax_valid is False
    assert "no such table" in (missing_table.validation_error or "")
