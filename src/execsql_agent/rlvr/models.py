"""Strict, verl-independent models for GRPO data and RLVR outcomes."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from execsql_agent.models import (
    ExecutionResult,
    ExpectedResult,
    SafetyCheckResult,
    StrictModel,
)


class PromptMessage(StrictModel):
    """One allowlisted model-visible prompt message."""

    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class DatabaseMetadata(StrictModel):
    """Trusted database identity recorded with an RLVR sample."""

    database_id: str = Field(min_length=1)
    database_path: str = Field(min_length=1)
    database_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    database_archive_path: str | None = None
    database_archive_member: str | None = None

    @model_validator(mode="after")
    def validate_archive_locator(self) -> Self:
        """Require the archive path and member together when either is present."""
        if (self.database_archive_path is None) != (
            self.database_archive_member is None
        ):
            raise ValueError(
                "database_archive_path and database_archive_member must be set together"
            )
        return self


class VerifierMetadata(StrictModel):
    """Private metadata available to the verifier but never to the model."""

    verifier_type: Literal["sqlite_execution_result_v1"] = "sqlite_execution_result_v1"
    response_contract: Literal[
        "single_sql_v1", "agentic_tool_loop_v1"
    ] = "single_sql_v1"
    expected_result: ExpectedResult


class ExtraInfo(StrictModel):
    """Non-secret provenance that is not required by the rollout prompt."""

    dataset_name: str = Field(min_length=1)
    difficulty: str | None = None
    evaluation_group: str | None = None
    tags: list[str] = Field(default_factory=list)
    prompt_version: Literal[
        "grpo_sql_json_v1", "grpo_agentic_harness_v2"
    ] = "grpo_sql_json_v1"
    comparator_version: Literal["execution_result_v1"] = "execution_result_v1"


class GRPOSample(StrictModel):
    """Intermediate training record intentionally independent of verl schemas."""

    sample_id: str = Field(min_length=1)
    data_source: str = Field(min_length=1)
    question: str = Field(min_length=1)
    messages: list[PromptMessage] = Field(min_length=2)
    database_metadata: DatabaseMetadata
    private_verifier_metadata: VerifierMetadata
    extra_info: ExtraInfo

    @model_validator(mode="after")
    def validate_prompt_boundary(self) -> Self:
        """Keep private verifier field names out of the model-visible messages."""

        if [message.role for message in self.messages] != ["system", "user"]:
            raise ValueError("messages must contain exactly one system and one user message")
        if self.messages[1].content != self.question:
            raise ValueError("the user message must contain only the sample question")
        prompt = "\n".join(message.content for message in self.messages).casefold()
        if "gold_sql" in prompt or "expected_result" in prompt:
            raise ValueError("private evaluation fields must not appear in messages")
        return self


class ParseStatus(StrEnum):
    """Whether one completion yielded exactly one reliable SQL candidate."""

    PARSED = "parsed"
    FAILED = "failed"


class ParseSource(StrEnum):
    """Unambiguous response representation used for SQL extraction."""

    JSON = "json"
    EXECUTE_SQL_TOOL_CALL = "execute_sql_tool_call"
    PLAIN_SQL = "plain_sql"


class ValidationStatus(StrEnum):
    """Static and SQLite compile-validation status."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    UNSAFE = "unsafe"
    FAILED = "failed"


class ExecutionStatus(StrEnum):
    """Read-only SQLite execution status."""

    NOT_RUN = "not_run"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ComparisonStatus(StrEnum):
    """Execution-result comparison status."""

    NOT_RUN = "not_run"
    MATCHED = "matched"
    MISMATCHED = "mismatched"


class VerifierFailureKind(StrEnum):
    """Stable failure reasons emitted without an LLM judge."""

    MALFORMED_RESPONSE = "malformed_response"
    MULTIPLE_SQL_CANDIDATES = "multiple_sql_candidates"
    NON_EXECUTE_SQL_TOOL = "non_execute_sql_tool"
    RESPONSE_DATABASE_PATH = "response_database_path"
    DATABASE_UNAVAILABLE = "database_unavailable"
    UNSAFE_SQL = "unsafe_sql"
    VALIDATION_FAILURE = "validation_failure"
    SQL_COMPILATION_FAILURE = "sql_compilation_failure"
    EXECUTION_FAILURE = "execution_failure"
    RESULT_MISMATCH = "result_mismatch"


class ResponseParseResult(StrictModel):
    """Structured SQL extraction result; failures never contain executable SQL."""

    parse_status: ParseStatus
    sql: str | None = None
    source: ParseSource | None = None
    failure_kind: VerifierFailureKind | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.parse_status is ParseStatus.PARSED:
            if not self.sql or self.source is None or self.failure_kind is not None:
                raise ValueError("a parsed response requires SQL and source only")
        elif self.sql is not None or self.source is not None or self.failure_kind is None:
            raise ValueError("a parse failure requires only a failure kind")
        return self


class VerifierResult(StrictModel):
    """Reward and complete deterministic diagnostics for one completion."""

    reward: float = Field(ge=0.0, le=1.0)
    parse_status: ParseStatus
    validation_status: ValidationStatus
    execution_status: ExecutionStatus
    comparison_status: ComparisonStatus
    failure_kind: VerifierFailureKind | None = None
    sql: str | None = None
    parse_source: ParseSource | None = None
    safety_check: SafetyCheckResult | None = None
    execution_result: ExecutionResult | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def validate_reward(self) -> Self:
        allowed = {0.0, 0.2, 1.0}
        if self.reward not in allowed:
            raise ValueError(f"reward must be one of {sorted(allowed)}")
        if self.reward == 1.0 and self.comparison_status is not ComparisonStatus.MATCHED:
            raise ValueError("reward 1.0 requires a matching result")
        if self.reward == 0.2 and self.comparison_status is not ComparisonStatus.MISMATCHED:
            raise ValueError("reward 0.2 requires an executable mismatching result")
        if self.reward == 0.0 and self.comparison_status is ComparisonStatus.MATCHED:
            raise ValueError("a matching result cannot receive zero reward")
        return self
