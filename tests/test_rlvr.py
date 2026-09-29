"""CPU-only tests for leak-resistant GRPO data and SQLite RLVR rewards."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from execsql_agent.evaluation.synthetic import load_evaluation_dataset
from execsql_agent.models import ExpectedResult
from execsql_agent.rlvr.models import (
    ComparisonStatus,
    ExecutionStatus,
    ParseSource,
    ParseStatus,
    ValidationStatus,
    VerifierFailureKind,
)
from execsql_agent.rlvr.response_parser import parse_sql_response
from execsql_agent.rlvr.verifier import SQLExecutionVerifier
from execsql_agent.tools.schema_loader import SchemaLoader
from training.build_grpo_dataset import build_samples


def _expected(
    columns: list[str],
    rows: list[list[object]],
    *,
    ordered: bool = True,
    numeric_tolerance: float = 1.0e-6,
) -> ExpectedResult:
    return ExpectedResult(
        columns=columns,
        rows=rows,
        ordered=ordered,
        numeric_tolerance=numeric_tolerance,
    )


def test_correct_sql_receives_one(demo_db: Path) -> None:
    result = SQLExecutionVerifier(demo_db).verify(
        '{"sql":"SELECT COUNT(*) AS customer_count FROM customers"}',
        _expected(["customer_count"], [[10]]),
    )
    assert result.reward == 1.0
    assert result.parse_status is ParseStatus.PARSED
    assert result.validation_status is ValidationStatus.PASSED
    assert result.execution_status is ExecutionStatus.SUCCEEDED
    assert result.comparison_status is ComparisonStatus.MATCHED
    assert result.failure_kind is None


def test_executable_wrong_result_receives_point_two(demo_db: Path) -> None:
    result = SQLExecutionVerifier(demo_db).verify(
        '{"sql":"SELECT COUNT(*) AS customer_count FROM products"}',
        _expected(["customer_count"], [[10]]),
    )
    assert result.reward == 0.2
    assert result.failure_kind is VerifierFailureKind.RESULT_MISMATCH


def test_syntax_error_receives_zero(demo_db: Path) -> None:
    result = SQLExecutionVerifier(demo_db).verify(
        '{"sql":"SELECT FROM customers"}', _expected([], [])
    )
    assert result.reward == 0.0
    assert result.validation_status is ValidationStatus.FAILED
    assert result.failure_kind is VerifierFailureKind.SQL_COMPILATION_FAILURE


def test_nonexistent_table_and_column_receive_zero(demo_db: Path) -> None:
    verifier = SQLExecutionVerifier(demo_db)
    missing_table = verifier.verify(
        '{"sql":"SELECT * FROM missing_table"}', _expected([], [])
    )
    missing_column = verifier.verify(
        '{"sql":"SELECT missing_column FROM customers"}', _expected([], [])
    )
    assert missing_table.reward == missing_column.reward == 0.0
    assert missing_table.failure_kind is VerifierFailureKind.SQL_COMPILATION_FAILURE
    assert missing_column.failure_kind is VerifierFailureKind.SQL_COMPILATION_FAILURE


def test_runtime_execution_failure_receives_zero(demo_db: Path) -> None:
    result = SQLExecutionVerifier(demo_db).verify(
        '{"sql":"SELECT abs(-9223372036854775808) AS value"}',
        _expected(["value"], [[0]]),
    )
    assert result.reward == 0.0
    assert result.validation_status is ValidationStatus.PASSED
    assert result.execution_status is ExecutionStatus.FAILED
    assert result.failure_kind is VerifierFailureKind.EXECUTION_FAILURE


def test_unsafe_sql_receives_zero_and_does_not_modify_database(demo_db: Path) -> None:
    result = SQLExecutionVerifier(demo_db).verify(
        '{"sql":"DELETE FROM customers"}', _expected([], [])
    )
    assert result.reward == 0.0
    assert result.validation_status is ValidationStatus.UNSAFE
    assert result.execution_status is ExecutionStatus.NOT_RUN
    assert result.failure_kind is VerifierFailureKind.UNSAFE_SQL
    with sqlite3.connect(demo_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM customers").fetchone() == (10,)


def test_malformed_json_and_natural_language_receive_zero(demo_db: Path) -> None:
    verifier = SQLExecutionVerifier(demo_db)
    malformed = verifier.verify('{"sql":"SELECT 1"', _expected([], []))
    prose = verifier.verify("Here is the SQL: SELECT 1", _expected([], []))
    assert malformed.reward == prose.reward == 0.0
    assert malformed.parse_status is ParseStatus.FAILED
    assert prose.failure_kind is VerifierFailureKind.MALFORMED_RESPONSE


def test_multiple_sql_candidates_receive_zero(demo_db: Path) -> None:
    result = SQLExecutionVerifier(demo_db).verify(
        '{"sql":"SELECT 1; SELECT 2"}', _expected([], [])
    )
    assert result.reward == 0.0
    assert result.parse_status is ParseStatus.FAILED
    assert result.failure_kind is VerifierFailureKind.MULTIPLE_SQL_CANDIDATES


def test_unordered_duplicate_rows_match(demo_db: Path) -> None:
    result = SQLExecutionVerifier(demo_db).verify(
        '{"sql":"SELECT 2 AS value UNION ALL SELECT 1 UNION ALL SELECT 1"}',
        _expected(["value"], [[1], [2], [1]], ordered=False),
    )
    assert result.reward == 1.0


def test_numeric_tolerance_matches(demo_db: Path) -> None:
    result = SQLExecutionVerifier(demo_db).verify(
        '{"sql":"SELECT 1.0005 AS value"}',
        _expected(["value"], [[1.0]], numeric_tolerance=0.001),
    )
    assert result.reward == 1.0


def test_truncated_execution_is_an_executable_mismatch(demo_db: Path) -> None:
    result = SQLExecutionVerifier(demo_db, max_rows=2).verify(
        '{"sql":"SELECT customer_id FROM customers ORDER BY customer_id"}',
        _expected(["customer_id"], [[1], [2]]),
    )
    assert result.execution_result is not None and result.execution_result.truncated
    assert result.reward == 0.2
    assert result.comparison_status is ComparisonStatus.MISMATCHED


def test_parser_accepts_reliable_tool_call_and_plain_sql() -> None:
    tool_call = parse_sql_response(
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "execute_sql",
                        "arguments": '{"sql":"SELECT 1"}',
                    },
                }
            ]
        }
    )
    plain = parse_sql_response("WITH value AS (SELECT 1 AS n) SELECT n FROM value")
    qwen = parse_sql_response(
        '<tool_call>{"name":"execute_sql","arguments":{"sql":"SELECT 1"}}</tool_call>'
    )
    assert tool_call.source is ParseSource.EXECUTE_SQL_TOOL_CALL
    assert plain.source is ParseSource.PLAIN_SQL
    assert qwen.source is ParseSource.EXECUTE_SQL_TOOL_CALL


def test_parser_rejects_non_execute_tool_and_database_path(demo_db: Path) -> None:
    wrong_tool = parse_sql_response(
        {"name": "inspect_schema", "arguments": {}}
    )
    malicious = SQLExecutionVerifier(demo_db).verify(
        '{"sql":"SELECT 1","database_path":"other.db"}',
        _expected(["1"], [[1]]),
    )
    assert wrong_tool.failure_kind is VerifierFailureKind.NON_EXECUTE_SQL_TOOL
    assert malicious.reward == 0.0
    assert malicious.failure_kind is VerifierFailureKind.RESPONSE_DATABASE_PATH
    assert malicious.execution_status is ExecutionStatus.NOT_RUN


def test_grpo_projection_omits_gold_and_keeps_expected_private(demo_db: Path) -> None:
    dataset = load_evaluation_dataset("data/synthetic/eval_questions.json")
    schema_context = SchemaLoader(demo_db).load().summary_text
    domain_context = "Test-only domain context."
    samples = build_samples(
        dataset,
        database_path=demo_db,
        database_sha256="0" * 64,
        schema_context=schema_context,
        domain_context=domain_context,
    )
    assert len(samples) == len(dataset.cases)
    for sample, case in zip(samples, dataset.cases, strict=True):
        serialized = sample.model_dump_json()
        messages = json.dumps(
            [message.model_dump(mode="json") for message in sample.messages],
            ensure_ascii=False,
        )
        assert "gold_sql" not in serialized
        assert "gold_sql" not in messages
        assert "expected_result" not in messages
        assert case.gold_sql is not None and case.gold_sql not in messages
        assert case.expected_result.model_dump_json() not in messages
        assert sample.private_verifier_metadata.expected_result == case.expected_result
        assert sample.question == case.question
        assert schema_context in sample.messages[0].content
        assert domain_context in sample.messages[0].content


def test_database_absence_is_reported_without_fabricating_results(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.db"
    result = SQLExecutionVerifier(missing).verify(
        '{"sql":"SELECT 1"}', _expected(["1"], [[1]])
    )
    assert result.reward == 0.0
    assert result.failure_kind is VerifierFailureKind.DATABASE_UNAVAILABLE
