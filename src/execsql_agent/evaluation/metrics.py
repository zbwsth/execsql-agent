"""Metric aggregation from persisted per-case evaluation facts."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from statistics import median
from typing import Literal

from execsql_agent.models import (
    BirdEXMetrics,
    CaseEvaluation,
    FailureKind,
    MetricValue,
    ModeComparison,
    ModeMetrics,
    SQLRuntimeMetrics,
    ToolMetrics,
)


def metric(numerator: float, denominator: int) -> MetricValue:
    """Build a metric with the required null-on-zero behavior."""

    return MetricValue(
        numerator=numerator,
        denominator=denominator,
        value=None if denominator == 0 else numerator / denominator,
    )


def _tool_metrics(cases: Sequence[CaseEvaluation]) -> ToolMetrics:
    annotated_sequences = [case for case in cases if case.expected_tool_sequences]
    exact_matches = sum(
        case.actual_tool_sequence in case.expected_tool_sequences
        for case in annotated_sequences
    )
    coverage_cases = [case for case in cases if case.required_tools]
    coverage_sum = 0.0
    for case in coverage_cases:
        actual = set(case.actual_tool_sequence)
        coverage_sum += len(actual.intersection(case.required_tools)) / len(
            set(case.required_tools)
        )

    total_calls = sum(case.total_tool_calls for case in cases)
    valid_arguments = sum(
        sum(case.tool_argument_validity) for case in cases
    )
    executed_calls = sum(len(case.tool_execution_successes) for case in cases)
    successful_calls = sum(
        sum(case.tool_execution_successes) for case in cases
    )
    invalid_calls = sum(case.invalid_tool_calls for case in cases)
    repeated_calls = sum(case.repeated_tool_calls for case in cases)
    return ToolMetrics(
        tool_sequence_exact_match_rate=metric(exact_matches, len(annotated_sequences)),
        required_tool_coverage=metric(coverage_sum, len(coverage_cases)),
        valid_tool_argument_rate=metric(valid_arguments, total_calls),
        tool_execution_success_rate=metric(successful_calls, executed_calls),
        invalid_tool_call_rate=metric(invalid_calls, total_calls),
        repeated_tool_call_rate=metric(repeated_calls, total_calls),
        average_tool_calls=metric(total_calls, len(cases)),
    )


def _bird_ex_metrics(
    cases: Sequence[CaseEvaluation],
    *,
    metric_kind: Literal["stable", "30s"],
) -> BirdEXMetrics | None:
    bird_cases = [case for case in cases if case.comparison_mode == "bird_set"]
    if not bird_cases:
        return None

    def accuracy(selected: Sequence[CaseEvaluation]) -> MetricValue:
        if metric_kind == "stable":
            numerator = sum(
                (
                    case.stable_bird_ex
                    if case.stable_bird_ex is not None
                    else case.result_correct
                )
                is True
                for case in selected
            )
        else:
            numerator = sum(case.ex_at_30s is True for case in selected)
        return metric(numerator, len(selected))

    return BirdEXMetrics(
        simple=accuracy([case for case in bird_cases if case.difficulty == "simple"]),
        moderate=accuracy(
            [case for case in bird_cases if case.difficulty == "moderate"]
        ),
        challenging=accuracy(
            [case for case in bird_cases if case.difficulty == "challenging"]
        ),
        overall=accuracy(bird_cases),
    )


def _prediction_sql_runtime_metrics(
    cases: Sequence[CaseEvaluation],
) -> SQLRuntimeMetrics | None:
    bird_cases = [case for case in cases if case.comparison_mode == "bird_set"]
    if not bird_cases:
        return None
    measured = [
        case.prediction_runtime_ms
        for case in bird_cases
        if case.prediction_runtime_ms is not None
    ]
    sorted_measured = sorted(measured)
    timeout_count = sum(case.prediction_timeout for case in bird_cases)
    return SQLRuntimeMetrics(
        measured_count=len(measured),
        median_ms=median(sorted_measured) if sorted_measured else None,
        p95_ms=(
            sorted_measured[ceil(0.95 * len(sorted_measured)) - 1]
            if sorted_measured
            else None
        ),
        timeout_count=timeout_count,
        timeout_rate=metric(timeout_count, len(bird_cases)),
    )


def calculate_mode_metrics(
    cases: Sequence[CaseEvaluation], *, include_tool_metrics: bool
) -> ModeMetrics:
    """Aggregate one mode without consulting prompts, SQL text, or report formats."""

    total = len(cases)
    first_execution_successes = sum(
        case.first_execution_success is True for case in cases
    )
    verifiable = [case for case in cases if case.result_correct is not None]
    first_execution_failures = [
        case for case in cases if case.first_execution_success is False
    ]
    repaired = sum(case.repair_succeeded is True for case in first_execution_failures)
    semantic_mismatches = sum(
        case.failure_kind is FailureKind.SEMANTIC_MISMATCH for case in cases
    )
    answerable_cases = [
        case
        for case in cases
        if not case.expected_refusal and not case.expected_unsafe_sql
    ]
    refusal_correct = sum(
        case.refused == case.expected_refusal for case in cases
    )
    unsafe_cases = [case for case in cases if case.expected_unsafe_sql]
    stable_bird_ex = _bird_ex_metrics(cases, metric_kind="stable")
    return ModeMetrics(
        total_cases=metric(total, 1),
        execution_accuracy=metric(
            sum(case.execution_success for case in answerable_cases),
            len(answerable_cases),
        ),
        protocol_completion_rate=metric(
            sum(case.protocol_completed for case in cases), total
        ),
        first_execution_success_rate=metric(first_execution_successes, total),
        final_execution_success_rate=metric(
            sum(case.execution_success for case in cases), total
        ),
        result_accuracy=metric(
            sum(case.result_correct is True for case in verifiable), len(verifiable)
        ),
        grounded_answer_rate=metric(sum(case.answer_grounded for case in cases), total),
        repair_success_rate=metric(repaired, len(first_execution_failures)),
        refusal_accuracy=metric(refusal_correct, total),
        unsafe_sql_block_rate=metric(
            sum(case.unsafe_sql_count > 0 for case in unsafe_cases),
            len(unsafe_cases),
        ),
        semantic_mismatch_count=metric(semantic_mismatches, 1),
        unsafe_sql_rate=metric(sum(case.unsafe_sql_count > 0 for case in cases), total),
        repeated_sql_rate=metric(
            sum(case.failure_kind is FailureKind.REPEATED_SQL for case in cases), total
        ),
        average_llm_turns=metric(sum(case.total_llm_turns for case in cases), total),
        average_sql_executions=metric(
            sum(case.total_sql_executions for case in cases), total
        ),
        average_duration_ms=metric(
            sum(case.total_duration_ms for case in cases), total
        ),
        tool_metrics=_tool_metrics(cases) if include_tool_metrics else None,
        bird_ex=stable_bird_ex,
        stable_bird_ex=stable_bird_ex,
        ex_at_30s=_bird_ex_metrics(cases, metric_kind="30s"),
        prediction_sql_runtime=_prediction_sql_runtime_metrics(cases),
    )


def _difference(function_value: float | None, pipeline_value: float | None) -> float | None:
    if function_value is None or pipeline_value is None:
        return None
    return function_value - pipeline_value


def compare_modes(pipeline: ModeMetrics, function_calling: ModeMetrics) -> ModeComparison:
    """Return Function Calling minus Pipeline differences."""

    return ModeComparison(
        result_accuracy_difference=_difference(
            function_calling.result_accuracy.value, pipeline.result_accuracy.value
        ),
        final_execution_success_rate_difference=_difference(
            function_calling.final_execution_success_rate.value,
            pipeline.final_execution_success_rate.value,
        ),
        repair_success_rate_difference=_difference(
            function_calling.repair_success_rate.value,
            pipeline.repair_success_rate.value,
        ),
        average_sql_executions_difference=_difference(
            function_calling.average_sql_executions.value,
            pipeline.average_sql_executions.value,
        ),
        average_duration_ms_difference=_difference(
            function_calling.average_duration_ms.value,
            pipeline.average_duration_ms.value,
        ),
    )
