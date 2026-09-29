"""Deterministically rescore persisted evaluation facts without calling an LLM."""

from __future__ import annotations

from execsql_agent.evaluation.comparator import compare_execution_result
from execsql_agent.evaluation.metrics import calculate_mode_metrics, compare_modes
from execsql_agent.models import (
    CaseEvaluation,
    EvaluationReport,
    FailureKind,
    ModeMetrics,
)


def _rescore_case(case: CaseEvaluation) -> CaseEvaluation:
    result_correct = compare_execution_result(case.actual_result, case.expected_result)
    failure_kind = case.failure_kind
    if result_correct is False:
        failure_kind = FailureKind.SEMANTIC_MISMATCH
    elif case.result_correct is False and case.failure_kind is FailureKind.SEMANTIC_MISMATCH:
        failure_kind = None
    return case.model_copy(
        update={
            "result_correct": result_correct,
            "failure_kind": failure_kind,
        }
    )


def rescore_evaluation_report(report: EvaluationReport) -> EvaluationReport:
    """Recompute correctness, failure classification, and metrics from saved results."""

    cases = [_rescore_case(case) for case in report.cases]
    metrics: dict[str, ModeMetrics] = {}
    for mode in report.mode_metrics:
        mode_cases = [case for case in cases if case.agent_mode == mode]
        metrics[mode] = calculate_mode_metrics(
            mode_cases, include_tool_metrics=mode == "function-calling"
        )
    comparison = None
    if "pipeline" in metrics and "function-calling" in metrics:
        comparison = compare_modes(metrics["pipeline"], metrics["function-calling"])
    return report.model_copy(
        update={
            "cases": cases,
            "mode_metrics": metrics,
            "comparison": comparison,
        }
    )
