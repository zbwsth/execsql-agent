"""Deterministic execution-result verifier for CPU-side RLVR rewards."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from execsql_agent.evaluation.comparator import compare_execution_result
from execsql_agent.models import ExpectedResult
from execsql_agent.rlvr.models import (
    ComparisonStatus,
    ExecutionStatus,
    ParseStatus,
    ValidationStatus,
    VerifierFailureKind,
    VerifierResult,
)
from execsql_agent.rlvr.response_parser import (
    parse_agentic_sql_response,
    parse_sql_response,
)
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator


class SQLExecutionVerifier:
    """Verify completions against one constructor-supplied trusted database path."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        max_rows: int = 100,
        validator: SQLValidator | None = None,
        executor: SQLExecutor | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.validator = validator or SQLValidator()
        self.executor = executor or SQLExecutor(
            self.database_path,
            max_rows=max_rows,
            validator=self.validator,
        )

    def verify(
        self,
        completion: str | Mapping[str, object],
        expected_result: ExpectedResult,
        *,
        response_contract: str = "single_sql_v1",
    ) -> VerifierResult:
        """Return a fixed 1.0/0.2/0.0 reward plus auditable diagnostics."""

        if response_contract not in {"single_sql_v1", "agentic_tool_loop_v1"}:
            raise ValueError(f"Unsupported response contract: {response_contract}")
        parsed = (
            parse_agentic_sql_response(completion)
            if response_contract == "agentic_tool_loop_v1"
            else parse_sql_response(completion)
        )
        if parsed.parse_status is ParseStatus.FAILED:
            return VerifierResult(
                reward=0.0,
                parse_status=parsed.parse_status,
                validation_status=ValidationStatus.NOT_RUN,
                execution_status=ExecutionStatus.NOT_RUN,
                comparison_status=ComparisonStatus.NOT_RUN,
                failure_kind=parsed.failure_kind,
                detail=parsed.detail,
            )

        sql = parsed.sql
        if sql is None:
            raise AssertionError("parsed SQL is unexpectedly missing")
        if not self.database_path.is_file():
            return VerifierResult(
                reward=0.0,
                parse_status=ParseStatus.PARSED,
                parse_source=parsed.source,
                sql=sql,
                validation_status=ValidationStatus.NOT_RUN,
                execution_status=ExecutionStatus.NOT_RUN,
                comparison_status=ComparisonStatus.NOT_RUN,
                failure_kind=VerifierFailureKind.DATABASE_UNAVAILABLE,
                detail=f"Trusted database does not exist: {self.database_path}",
            )

        safety = self.validator.validate(sql, database_path=self.database_path)
        if not safety.safe:
            failure_kind = (
                VerifierFailureKind.VALIDATION_FAILURE
                if safety.blocked_operation in {"EMPTY_SQL", "MALFORMED_SQL", "MULTI_STATEMENT"}
                else VerifierFailureKind.UNSAFE_SQL
            )
            return VerifierResult(
                reward=0.0,
                parse_status=ParseStatus.PARSED,
                parse_source=parsed.source,
                sql=sql,
                validation_status=(
                    ValidationStatus.FAILED
                    if failure_kind is VerifierFailureKind.VALIDATION_FAILURE
                    else ValidationStatus.UNSAFE
                ),
                execution_status=ExecutionStatus.NOT_RUN,
                comparison_status=ComparisonStatus.NOT_RUN,
                failure_kind=failure_kind,
                safety_check=safety,
                detail=safety.reason,
            )
        if safety.syntax_valid is not True:
            return VerifierResult(
                reward=0.0,
                parse_status=ParseStatus.PARSED,
                parse_source=parsed.source,
                sql=sql,
                validation_status=ValidationStatus.FAILED,
                execution_status=ExecutionStatus.NOT_RUN,
                comparison_status=ComparisonStatus.NOT_RUN,
                failure_kind=VerifierFailureKind.SQL_COMPILATION_FAILURE,
                safety_check=safety,
                detail=safety.validation_error,
            )

        execution = self.executor.execute(sql)
        if not execution.execution_success:
            detail = (
                execution.error.message
                if execution.error is not None
                else execution.blocked_reason
            )
            return VerifierResult(
                reward=0.0,
                parse_status=ParseStatus.PARSED,
                parse_source=parsed.source,
                sql=sql,
                validation_status=ValidationStatus.PASSED,
                execution_status=ExecutionStatus.FAILED,
                comparison_status=ComparisonStatus.NOT_RUN,
                failure_kind=VerifierFailureKind.EXECUTION_FAILURE,
                safety_check=safety,
                execution_result=execution,
                detail=detail,
            )

        comparison = compare_execution_result(execution, expected_result)
        if comparison is True:
            return VerifierResult(
                reward=1.0,
                parse_status=ParseStatus.PARSED,
                parse_source=parsed.source,
                sql=sql,
                validation_status=ValidationStatus.PASSED,
                execution_status=ExecutionStatus.SUCCEEDED,
                comparison_status=ComparisonStatus.MATCHED,
                safety_check=safety,
                execution_result=execution,
            )
        if comparison is False:
            return VerifierResult(
                reward=0.2,
                parse_status=ParseStatus.PARSED,
                parse_source=parsed.source,
                sql=sql,
                validation_status=ValidationStatus.PASSED,
                execution_status=ExecutionStatus.SUCCEEDED,
                comparison_status=ComparisonStatus.MISMATCHED,
                failure_kind=VerifierFailureKind.RESULT_MISMATCH,
                safety_check=safety,
                execution_result=execution,
                detail="SQL executed successfully but its result did not match.",
            )
        return VerifierResult(
            reward=0.0,
            parse_status=ParseStatus.PARSED,
            parse_source=parsed.source,
            sql=sql,
            validation_status=ValidationStatus.PASSED,
            execution_status=ExecutionStatus.SUCCEEDED,
            comparison_status=ComparisonStatus.NOT_RUN,
            failure_kind=VerifierFailureKind.VALIDATION_FAILURE,
            safety_check=safety,
            execution_result=execution,
            detail="A successful execution could not be compared reliably.",
        )
