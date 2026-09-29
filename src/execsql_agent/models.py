"""Core Pydantic models shared across ExecSQL-Agent boundaries."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base data model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class ColumnSchema(StrictModel):
    """A column reported by SQLite schema introspection."""

    name: str
    data_type: str
    nullable: bool
    default_value: str | None = None
    primary_key: bool = False


class ForeignKeySchema(StrictModel):
    """A foreign-key edge between two SQLite tables."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str | None
    on_update: str
    on_delete: str


class TableSchema(StrictModel):
    """Structured metadata for one SQLite table."""

    name: str
    columns: list[ColumnSchema]
    primary_keys: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeySchema] = Field(default_factory=list)


class DatabaseSchema(StrictModel):
    """Structured and model-readable schema for one database."""

    database_id: str
    tables: list[TableSchema]
    summary_text: str


class SafetyCheckResult(StrictModel):
    """Result of static SQL safety and optional compile validation."""

    safe: bool
    normalized_sql: str
    reason: str | None = None
    blocked_operation: str | None = None
    syntax_valid: bool | None = None
    validation_error: str | None = None


class ExecutionError(StrictModel):
    """A concrete SQLite error raised while opening or executing a query."""

    error_type: str = "execution_error"
    message: str
    sqlite_error_name: str | None = None
    sqlite_error_code: int | None = None


class ExecutionResult(StrictModel):
    """A bounded result returned by the read-only SQL executor."""

    executed: bool
    execution_success: bool
    columns: list[str] = Field(default_factory=list)
    rows: list[list[object]] = Field(default_factory=list)
    returned_row_count: int = Field(default=0, ge=0)
    truncated: bool = False
    error: ExecutionError | None = None
    blocked_reason: str | None = None
    duration_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Prevent ambiguous execution state and misleading row counts."""

        if self.returned_row_count != len(self.rows):
            raise ValueError("returned_row_count must equal the number of returned rows")
        if self.execution_success and not self.executed:
            raise ValueError("a successful execution must have executed SQL")
        if self.execution_success and self.error is not None:
            raise ValueError("a successful execution cannot contain an error")
        if not self.executed and self.rows:
            raise ValueError("a non-executed query cannot return rows")
        return self


class GenerationMode(StrEnum):
    """Whether SQL is generated initially or as an error repair."""

    INITIAL_GENERATION = "initial_generation"
    REPAIR = "repair"


class SQLGeneration(StrictModel):
    """Validated SQL candidate produced by a language model."""

    sql: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    referenced_tables: list[str] = Field(default_factory=list)
    referenced_columns: list[str] = Field(default_factory=list)


class ErrorType(StrEnum):
    """Stable error taxonomy used by diagnosis and evaluation."""

    SYNTAX_ERROR = "syntax_error"
    MISSING_TABLE = "missing_table"
    MISSING_COLUMN = "missing_column"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    AGGREGATION_ERROR = "aggregation_error"
    INCORRECT_JOIN = "incorrect_join"
    EMPTY_RESULT = "empty_result"
    REPEATED_SQL = "repeated_sql"
    UNSAFE_SQL = "unsafe_sql"
    UNKNOWN_ERROR = "unknown_error"


class DiagnosisSource(StrEnum):
    """Origin of an error diagnosis."""

    RULE = "rule"
    LLM = "llm"


class ErrorDiagnosis(StrictModel):
    """Structured explanation and repair direction for a failed SQL candidate."""

    error_type: ErrorType
    cause: str = Field(min_length=1)
    repair_instruction: str = Field(min_length=1)
    related_tables: list[str] = Field(default_factory=list)
    related_columns: list[str] = Field(default_factory=list)
    source: DiagnosisSource


class LLMError(StrictModel):
    """Provider-independent model failure details."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    attempts: int = Field(default=1, ge=1)
    status_code: int | None = None


class ToolDefinition(StrictModel):
    """A model-visible function described with JSON Schema."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, object]


class ToolCallRequest(StrictModel):
    """A validated tool-call request parsed from a model response."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, object] | None = None
    raw_arguments: str | None = None


class ResponseMode(StrEnum):
    """Protocol used to represent the assistant response."""

    NATIVE_TOOL_CALLS = "native_tool_calls"
    JSON_FALLBACK = "json_fallback"
    PLAIN_FINAL = "plain_final"


class LLMResponse(StrictModel):
    """Normalized response from any language-model backend."""

    final_answer: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str | None = None
    response_mode: ResponseMode
    raw_content: str | None = None

    @model_validator(mode="after")
    def require_output(self) -> Self:
        """Reject incomplete responses that contain neither text nor calls."""

        if not self.tool_calls and not self.final_answer:
            raise ValueError("LLMResponse requires a final answer or at least one tool call")
        return self


class LLMMessage(StrictModel):
    """Provider-neutral chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_role_shape(self) -> Self:
        """Enforce OpenAI-compatible role and tool-call consistency."""

        if self.role in {"system", "user"}:
            if not self.content:
                raise ValueError(f"{self.role} messages require content")
            if self.tool_call_id is not None or self.tool_calls:
                raise ValueError(f"{self.role} messages cannot contain tool metadata")
        elif self.role == "assistant":
            if self.tool_call_id is not None:
                raise ValueError("assistant messages cannot contain tool_call_id")
            if not self.content and not self.tool_calls:
                raise ValueError("assistant messages require content or tool_calls")
        elif self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
            if self.content is None:
                raise ValueError("tool messages require content")
            if self.tool_calls:
                raise ValueError("tool messages cannot contain tool_calls")
        return self


class LLMCallConfig(StrictModel):
    """Options for one provider-neutral model call."""

    temperature: float = Field(default=0.0, ge=0)
    max_tokens: int = Field(default=1_000, ge=1)
    response_schema_name: str = "structured_response"
    response_schema: dict[str, object] | None = None


class LLMRequest(StrictModel):
    """Complete provider-neutral input to an LLM client."""

    messages: list[LLMMessage] = Field(min_length=1)
    tools: list[ToolDefinition] = Field(default_factory=list)
    config: LLMCallConfig = Field(default_factory=LLMCallConfig)


class PipelineTerminationReason(StrEnum):
    """Explicit terminal condition for the fixed Pipeline Agent."""

    COMPLETED = "completed"
    MAX_STEPS_REACHED = "max_steps_reached"
    REPEATED_SQL = "repeated_sql"
    UNSAFE_SQL = "unsafe_sql"
    MODEL_ERROR = "model_error"
    UNRECOVERABLE_ERROR = "unrecoverable_error"


class PipelineStep(StrictModel):
    """One complete SQL generation or repair attempt."""

    step_type: Literal["pipeline"] = "pipeline"
    step_index: int = Field(ge=1)
    mode: GenerationMode
    generation: SQLGeneration | None = None
    safety: SafetyCheckResult | None = None
    execution: ExecutionResult | None = None
    diagnosis: ErrorDiagnosis | None = None
    llm_error: LLMError | None = None
    observation: str


class PipelineAgentResult(StrictModel):
    """Final Pipeline output with deliberately separate success semantics."""

    question: str
    database_id: str
    schema_summary: str
    steps: list[PipelineStep]
    final_sql: str | None = None
    execution_result: ExecutionResult | None = None
    protocol_completed: bool
    execution_success: bool
    answer_grounded: bool
    result_correct: bool | None = None
    termination_reason: PipelineTerminationReason
    total_steps: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_statuses(self) -> Self:
        """Keep protocol, execution, grounding, and correctness independent."""

        if self.total_steps != len(self.steps):
            raise ValueError("total_steps must equal the number of recorded steps")
        if self.result_correct is not None:
            raise ValueError("PipelineAgent does not determine result_correct")
        if self.execution_success:
            if self.execution_result is None or not self.execution_result.execution_success:
                raise ValueError("execution_success requires a successful ExecutionResult")
        if self.answer_grounded and not self.execution_success:
            raise ValueError("answer_grounded requires execution_success")
        return self


class ToolCallValidation(StrictModel):
    """Name, argument, and repeat validation for one requested tool call."""

    tool_call_id: str
    tool_name: str
    valid: bool
    known_tool: bool
    arguments_valid: bool
    duplicate: bool = False
    canonical_signature: str
    error_code: str | None = None
    error_message: str | None = None


class ToolCallResult(StrictModel):
    """Structured outcome of validation and optional tool execution."""

    tool_call_id: str
    tool_name: str
    success: bool
    executed: bool
    output: dict[str, object] | None = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float = Field(ge=0)


class ToolMessage(StrictModel):
    """Tool observation linked to the assistant request that caused it."""

    role: Literal["tool"] = "tool"
    tool_call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    content: str

    def to_llm_message(self) -> LLMMessage:
        """Convert the observation to provider-neutral message history."""

        return LLMMessage(
            role="tool", tool_call_id=self.tool_call_id, content=self.content
        )


class FunctionCallingTerminationReason(StrEnum):
    """Explicit terminal condition for FunctionCallingAgent."""

    COMPLETED = "completed"
    MAX_STEPS_REACHED = "max_steps_reached"
    UNRESOLVED_COMPLETION = "unresolved_completion"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    REPEATED_SQL = "repeated_sql"
    UNSAFE_SQL = "unsafe_sql"
    MODEL_ERROR = "model_error"
    UNRECOVERABLE_ERROR = "unrecoverable_error"


class CompletionKind(StrEnum):
    """How an agent run produced or failed to produce its final answer."""

    DIRECT_ANSWER = "direct_answer"
    TOOL_GROUNDED_ANSWER = "tool_grounded_answer"
    TERMINATED = "terminated"


class FunctionCallingStep(StrictModel):
    """One LLM turn and all ordered tool observations caused by it."""

    step_type: Literal["function_calling"] = "function_calling"
    turn_index: int = Field(ge=1)
    input_message_summary: list[str]
    completion_guard_triggered: bool = False
    completion_guard_instruction: str | None = None
    llm_response: LLMResponse | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    validations: list[ToolCallValidation] = Field(default_factory=list)
    tool_results: list[ToolCallResult] = Field(default_factory=list)
    tool_messages: list[ToolMessage] = Field(default_factory=list)
    llm_error: LLMError | None = None
    duration_ms: float = Field(ge=0)


class FunctionCallingAgentResult(StrictModel):
    """Function Calling result with independent completion semantics."""

    trajectory_id: str
    session_id: str = "default"
    question: str
    database_id: str
    schema_summary: str
    steps: list[FunctionCallingStep]
    final_answer: str | None = None
    final_sql: str | None = None
    execution_result: ExecutionResult | None = None
    protocol_completed: bool
    execution_success: bool
    answer_grounded: bool
    result_correct: bool | None = None
    completion_kind: CompletionKind
    termination_reason: FunctionCallingTerminationReason
    total_llm_turns: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    sql_execution_count: int = Field(ge=0)
    total_duration_ms: float = Field(ge=0)
    trajectory_path: str | None = None

    @model_validator(mode="after")
    def validate_statuses(self) -> Self:
        """Reject mixed or overstated completion semantics."""

        if self.total_llm_turns != len(self.steps):
            raise ValueError("total_llm_turns must equal recorded Function Calling steps")
        if self.result_correct is not None:
            raise ValueError("FunctionCallingAgent does not determine result_correct")
        if self.answer_grounded and not self.execution_success:
            raise ValueError("answer_grounded requires execution_success")
        if self.protocol_completed and not self.final_answer:
            raise ValueError("protocol_completed requires a final answer")
        if self.completion_kind is CompletionKind.DIRECT_ANSWER:
            if self.execution_success or self.answer_grounded:
                raise ValueError("direct answers cannot claim execution grounding")
        if self.completion_kind is CompletionKind.TOOL_GROUNDED_ANSWER:
            if not (self.protocol_completed and self.execution_success and self.answer_grounded):
                raise ValueError("tool-grounded answers require completed successful execution")
        return self


StepRecord = Annotated[
    PipelineStep | FunctionCallingStep,
    Field(discriminator="step_type"),
]


class Trajectory(StrictModel):
    """Versioned, reloadable record for one complete agent task."""

    trajectory_id: str
    session_id: str | None = None
    schema_version: str = "1.0"
    agent_mode: Literal["pipeline", "function-calling"]
    question: str
    database_id: str
    schema_summary: str
    llm_backend: str
    run_mode: str
    steps: list[StepRecord]
    final_answer: str | None = None
    final_sql: str | None = None
    execution_result: ExecutionResult | None = None
    protocol_completed: bool
    execution_success: bool
    answer_grounded: bool
    result_correct: bool | None = None
    completion_kind: CompletionKind
    termination_reason: str
    total_llm_turns: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    sql_execution_count: int = Field(ge=0)
    total_tool_calls: int = Field(default=0, ge=0)
    total_sql_executions: int = Field(default=0, ge=0)
    total_duration_ms: float = Field(ge=0)
    dataset_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    message_history: list[LLMMessage] = Field(default_factory=list)
    assistant_tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_observations: list[ToolMessage] = Field(default_factory=list)
    invalid_tool_calls: int = Field(default=0, ge=0)
    repeated_tool_calls: int = Field(default=0, ge=0)
    unsafe_sql_count: int = Field(default=0, ge=0)


class SessionMemoryTurn(StrictModel):
    """Bounded summary of one completed Function Calling conversation turn."""

    question: str
    generated_sql: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    execution_summary: str
    final_answer: str | None = None
    termination_reason: str


class EvaluationAgentMode(StrEnum):
    """Agent modes accepted by the batch evaluator."""

    PIPELINE = "pipeline"
    FUNCTION_CALLING = "function-calling"
    BOTH = "both"


class FailureKind(StrEnum):
    """Evaluation-level failure categories, separate from SQLite errors."""

    EXECUTION_ERROR = "execution_error"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    UNSAFE_SQL = "unsafe_sql"
    REPEATED_SQL = "repeated_sql"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    INVALID_TOOL_CALL = "invalid_tool_call"
    UNGROUNDED_ANSWER = "ungrounded_answer"
    MAX_STEPS = "max_steps"
    UNRESOLVED_COMPLETION = "unresolved_completion"
    MODEL_ERROR = "model_error"
    UNRECOVERABLE_ERROR = "unrecoverable_error"


class ExpectedResult(StrictModel):
    """Expected query output and its comparison policy."""

    columns: list[str]
    rows: list[list[object]]
    ordered: bool = True
    numeric_tolerance: float = Field(default=1.0e-6, ge=0)
    strict_columns: bool = False


class EvaluationCase(StrictModel):
    """One normal Text-to-SQL evaluation case."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    database_id: str = Field(min_length=1)
    expected_result: ExpectedResult
    gold_sql: str | None = None
    gold_runtime_ms: float | None = Field(default=None, ge=0)
    difficulty: str | None = None
    evidence: str | None = None
    evaluation_group: str | None = None
    expected_query_shape: str | None = None
    comparison_mode: str = "result"
    notes: str | None = None
    expected_refusal: bool = False
    expected_unsafe_sql: bool = False
    required_tables: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    expected_tool_sequences: list[list[str]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    fake_sql: str | None = None
    fake_responses: dict[str, list[LLMResponse]] | None = None


class BehaviorScenario(EvaluationCase):
    """Deterministic FakeLLM behavior case with per-mode response scripts."""

    fake_responses: dict[str, list[LLMResponse]]
    max_steps: dict[str, int] = Field(default_factory=dict)
    follow_up_question: str | None = None
    follow_up_responses: list[LLMResponse] = Field(default_factory=list)


class EvaluationDataset(StrictModel):
    """Named collection of normal evaluation cases."""

    dataset_name: str
    cases: list[EvaluationCase]


class BehaviorDataset(StrictModel):
    """Named collection of deterministic protocol and recovery scenarios."""

    dataset_name: str
    cases: list[BehaviorScenario]


class MetricValue(StrictModel):
    """Auditable metric represented by its numerator, denominator, and value."""

    numerator: float
    denominator: int = Field(ge=0)
    value: float | None

    @model_validator(mode="after")
    def validate_ratio(self) -> Self:
        """Require null for a zero denominator and the exact computed ratio otherwise."""

        if self.denominator == 0:
            if self.value is not None:
                raise ValueError("zero-denominator metrics must have value=None")
            return self
        expected = self.numerator / self.denominator
        if self.value is None or abs(self.value - expected) > 1.0e-12:
            raise ValueError("metric value must equal numerator / denominator")
        return self


class ToolMetrics(StrictModel):
    """Function Calling metrics derived from raw requested tool calls."""

    tool_sequence_exact_match_rate: MetricValue
    required_tool_coverage: MetricValue
    valid_tool_argument_rate: MetricValue
    tool_execution_success_rate: MetricValue
    invalid_tool_call_rate: MetricValue
    repeated_tool_call_rate: MetricValue
    average_tool_calls: MetricValue


class BirdEXMetrics(StrictModel):
    """Official BIRD execution accuracy overall and by difficulty."""

    simple: MetricValue
    moderate: MetricValue
    challenging: MetricValue
    overall: MetricValue


class BirdPredictionTimeoutPolicy(StrictModel):
    """Per-case BIRD prediction deadline derived from cached gold runtime."""

    minimum_seconds: float = Field(default=30.0, gt=0)
    gold_runtime_multiplier: float = Field(default=1.5, ge=0)
    offset_seconds: float = Field(default=5.0, ge=0)
    upper_cap_seconds: float = Field(default=900.0, gt=0)
    ex_at_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.upper_cap_seconds < self.minimum_seconds:
            raise ValueError("upper_cap_seconds must be >= minimum_seconds")
        return self

    def timeout_seconds(self, gold_runtime_ms: float) -> float:
        derived = (
            self.gold_runtime_multiplier * (gold_runtime_ms / 1000)
            + self.offset_seconds
        )
        return min(
            self.upper_cap_seconds,
            max(self.minimum_seconds, derived),
        )


class BirdExperimentProtocol(StrictModel):
    """Frozen settings shared by BIRD Base and SFT evaluation runs."""

    schema_version: str = "1.0"
    dataset_revision: str
    agent_mode: Literal["function-calling"] = "function-calling"
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    max_tokens: int = Field(default=2048, ge=1)
    seed: int = 0
    enable_thinking: bool = False
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_agent_steps: int = Field(default=5, ge=1)
    use_evidence: bool = False
    timeout_policy: BirdPredictionTimeoutPolicy = Field(
        default_factory=BirdPredictionTimeoutPolicy
    )
    vllm_common: dict[str, object]


class SQLRuntimeMetrics(StrictModel):
    """Auxiliary scorer-side prediction SQL runtime statistics."""

    measured_count: int = Field(ge=0)
    median_ms: float | None = Field(default=None, ge=0)
    p95_ms: float | None = Field(default=None, ge=0)
    timeout_count: int = Field(ge=0)
    timeout_rate: MetricValue


class ModeMetrics(StrictModel):
    """All metrics for one Agent mode over one selected case set."""

    total_cases: MetricValue
    execution_accuracy: MetricValue
    protocol_completion_rate: MetricValue
    first_execution_success_rate: MetricValue
    final_execution_success_rate: MetricValue
    result_accuracy: MetricValue
    grounded_answer_rate: MetricValue
    repair_success_rate: MetricValue
    refusal_accuracy: MetricValue
    unsafe_sql_block_rate: MetricValue
    semantic_mismatch_count: MetricValue
    unsafe_sql_rate: MetricValue
    repeated_sql_rate: MetricValue
    average_llm_turns: MetricValue
    average_sql_executions: MetricValue
    average_duration_ms: MetricValue
    tool_metrics: ToolMetrics | None = None
    bird_ex: BirdEXMetrics | None = None
    stable_bird_ex: BirdEXMetrics | None = None
    ex_at_30s: BirdEXMetrics | None = None
    prediction_sql_runtime: SQLRuntimeMetrics | None = None


class CaseEvaluation(StrictModel):
    """Auditable outcome for one case and one Agent mode."""

    case_id: str
    dataset_name: str
    tags: list[str]
    question: str
    database_id: str
    agent_mode: Literal["pipeline", "function-calling"]
    protocol_completed: bool
    execution_success: bool
    answer_grounded: bool
    result_correct: bool | None
    difficulty: str | None = None
    comparison_mode: str = "result"
    expected_refusal: bool = False
    refused: bool = False
    expected_unsafe_sql: bool = False
    failure_kind: FailureKind | None = None
    termination_reason: str
    final_sql: str | None = None
    final_answer: str | None = None
    expected_result: ExpectedResult
    gold_sql: str | None = None
    gold_runtime_ms: float | None = Field(default=None, ge=0)
    prediction_timeout_budget_seconds: float | None = Field(default=None, gt=0)
    prediction_runtime_ms: float | None = Field(default=None, ge=0)
    prediction_timeout: bool = False
    stable_bird_ex: bool | None = None
    ex_at_30s: bool | None = None
    required_tools: list[str] = Field(default_factory=list)
    expected_tool_sequences: list[list[str]] = Field(default_factory=list)
    actual_result: ExecutionResult | None = None
    first_execution_success: bool | None = None
    repair_succeeded: bool | None = None
    trajectory_id: str
    total_duration_ms: float = Field(ge=0)
    total_llm_turns: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    total_sql_executions: int = Field(ge=0)
    invalid_tool_calls: int = Field(ge=0)
    repeated_tool_calls: int = Field(ge=0)
    unsafe_sql_count: int = Field(ge=0)
    message_history: list[LLMMessage] = Field(default_factory=list)
    assistant_tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_observations: list[ToolMessage] = Field(default_factory=list)
    actual_tool_sequence: list[str] = Field(default_factory=list)
    tool_argument_validity: list[bool] = Field(default_factory=list)
    tool_execution_successes: list[bool] = Field(default_factory=list)
    error_message: str | None = None


class EvaluationCheckpointRecord(StrictModel):
    """One fully evaluated case persisted for config-safe resume."""

    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str
    agent_mode: Literal["pipeline", "function-calling"]
    evaluation: CaseEvaluation


class ModeComparison(StrictModel):
    """Function Calling minus Pipeline metric differences on identical cases."""

    result_accuracy_difference: float | None
    final_execution_success_rate_difference: float | None
    repair_success_rate_difference: float | None
    average_sql_executions_difference: float | None
    average_duration_ms_difference: float | None


class EvaluationReport(StrictModel):
    """Single source of truth for JSON, CSV, and Markdown reports."""

    schema_version: str = "1.0"
    dataset_name: str
    database_id: str
    agent_mode: EvaluationAgentMode
    llm_backend: str
    run_mode: str
    seed: int
    database_ids: list[str] = Field(default_factory=list)
    oracle_evidence: bool = False
    experiment_config_fingerprint: str | None = None
    selected_tags: list[str] = Field(default_factory=list)
    cases: list[CaseEvaluation]
    mode_metrics: dict[str, ModeMetrics]
    comparison: ModeComparison | None = None
