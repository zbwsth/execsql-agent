"""Tests for deterministic SQLite error diagnosis."""

import json
from pathlib import Path

import pytest

from execsql_agent.diagnosis.error_diagnoser import ErrorDiagnoser
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import (
    DiagnosisSource,
    ErrorType,
    LLMResponse,
    ResponseMode,
)
from execsql_agent.tools.schema_loader import SchemaLoader


def test_missing_table_rule_suggests_real_table(demo_db: Path) -> None:
    diagnosis = ErrorDiagnoser().diagnose(
        sql="SELECT * FROM customer",
        schema=SchemaLoader(demo_db).load(),
        error_message="no such table: customer",
    )

    assert diagnosis.error_type is ErrorType.MISSING_TABLE
    assert diagnosis.source is DiagnosisSource.RULE
    assert "customers" in diagnosis.related_tables


def test_missing_column_rule_suggests_real_column(demo_db: Path) -> None:
    diagnosis = ErrorDiagnoser().diagnose(
        sql="SELECT customer_nam FROM customers",
        schema=SchemaLoader(demo_db).load(),
        error_message="no such column: customer_nam",
    )

    assert diagnosis.error_type is ErrorType.MISSING_COLUMN
    assert diagnosis.source is DiagnosisSource.RULE
    assert "customer_name" in diagnosis.related_columns
    assert "customers" in diagnosis.related_tables


@pytest.mark.parametrize(
    ("message", "expected_type"),
    [
        ('near "FROM": syntax error', ErrorType.SYNTAX_ERROR),
        ("ambiguous column name: customer_id", ErrorType.AMBIGUOUS_COLUMN),
    ],
)
def test_other_explicit_sqlite_rules(
    demo_db: Path, message: str, expected_type: ErrorType
) -> None:
    diagnosis = ErrorDiagnoser().diagnose(
        sql="SELECT customer_id",
        schema=SchemaLoader(demo_db).load(),
        error_message=message,
    )

    assert diagnosis.error_type is expected_type
    assert diagnosis.source is DiagnosisSource.RULE


def test_forced_unsafe_and_repeated_rules_do_not_call_llm(demo_db: Path) -> None:
    fake = FakeLLMClient([])
    diagnoser = ErrorDiagnoser(fake)
    schema = SchemaLoader(demo_db).load()

    unsafe = diagnoser.diagnose(
        sql="DELETE FROM customers",
        schema=schema,
        error_message="DELETE is blocked",
        forced_error_type=ErrorType.UNSAFE_SQL,
    )
    repeated = diagnoser.diagnose(
        sql="SELECT 1",
        schema=schema,
        error_message="repeated",
        forced_error_type=ErrorType.REPEATED_SQL,
    )

    assert unsafe.error_type is ErrorType.UNSAFE_SQL
    assert repeated.error_type is ErrorType.REPEATED_SQL
    assert fake.requests == []


def test_unknown_error_can_use_llm_fallback(demo_db: Path) -> None:
    response = LLMResponse(
        final_answer=json.dumps(
            {
                "error_type": "unknown_error",
                "cause": "provider analysis",
                "repair_instruction": "rewrite conservatively",
                "related_tables": [],
                "related_columns": [],
                "source": "llm",
            }
        ),
        response_mode=ResponseMode.PLAIN_FINAL,
    )
    fake = FakeLLMClient([response])

    diagnosis = ErrorDiagnoser(fake).diagnose(
        sql="SELECT 1",
        schema=SchemaLoader(demo_db).load(),
        error_message="opaque provider-specific failure",
    )

    assert diagnosis.error_type is ErrorType.UNKNOWN_ERROR
    assert diagnosis.source is DiagnosisSource.LLM
    assert len(fake.requests) == 1
