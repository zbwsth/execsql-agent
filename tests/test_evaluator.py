"""Synthetic evaluation, result comparison, metrics, and report tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from execsql_agent.evaluation.comparator import compare_execution_result
from execsql_agent.evaluation.evaluator import Evaluator
from execsql_agent.evaluation.reports import write_reports
from execsql_agent.evaluation.rescore import rescore_evaluation_report
from execsql_agent.evaluation.synthetic import (
    build_scripted_client,
    load_behavior_dataset,
    load_evaluation_dataset,
)
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import (
    EvaluationAgentMode,
    EvaluationCase,
    EvaluationReport,
    ExecutionResult,
    ExpectedResult,
    FailureKind,
    LLMResponse,
    ResponseMode,
)


def _execution(
    rows: list[list[object]],
    columns: list[str] | None = None,
    *,
    truncated: bool = False,
) -> ExecutionResult:
    return ExecutionResult(
        executed=True,
        execution_success=True,
        columns=columns or ["value"],
        rows=rows,
        returned_row_count=len(rows),
        truncated=truncated,
        duration_ms=0,
    )


def test_ordered_result_comparison() -> None:
    expected = ExpectedResult(columns=["value"], rows=[[1], [2]], ordered=True)
    assert compare_execution_result(_execution([[1], [2]]), expected) is True
    assert compare_execution_result(_execution([[2], [1]]), expected) is False


def test_unordered_result_preserves_duplicate_rows() -> None:
    expected = ExpectedResult(
        columns=["value"], rows=[[1], [1], [2]], ordered=False
    )
    assert compare_execution_result(_execution([[2], [1], [1]]), expected) is True
    assert compare_execution_result(_execution([[2], [2], [1]]), expected) is False


def test_null_and_numeric_tolerance() -> None:
    expected = ExpectedResult(
        columns=["name", "score"],
        rows=[[None, 1.0]],
        numeric_tolerance=0.01,
    )
    assert compare_execution_result(
        _execution([[None, 1.005]], ["name", "score"]), expected
    ) is True
    assert compare_execution_result(
        _execution([["", 1.005]], ["name", "score"]), expected
    ) is False


def test_strict_and_relaxed_column_comparison() -> None:
    relaxed = ExpectedResult(columns=["expected"], rows=[[1]])
    strict = ExpectedResult(columns=["expected"], rows=[[1]], strict_columns=True)
    actual = _execution([[1]], ["actual"])
    assert compare_execution_result(actual, strict) is False
    assert compare_execution_result(actual, relaxed) is True


def test_relaxed_columns_still_require_position_and_compatible_types() -> None:
    numeric = ExpectedResult(columns=["expected"], rows=[[1.0]])
    text = ExpectedResult(columns=["expected"], rows=[["1"]])
    boolean = ExpectedResult(columns=["expected"], rows=[[True]])

    assert compare_execution_result(_execution([[1]], ["alias"]), numeric) is True
    assert compare_execution_result(_execution([[1]], ["alias"]), text) is False
    assert compare_execution_result(_execution([[1]], ["alias"]), boolean) is False
    assert (
        compare_execution_result(_execution([[1, 2]], ["a", "b"]), numeric)
        is False
    )


def test_empty_result_is_valid() -> None:
    expected = ExpectedResult(columns=["value"], rows=[])
    assert compare_execution_result(_execution([]), expected) is True


def test_truncated_successful_result_is_incorrect() -> None:
    expected = ExpectedResult(columns=["value"], rows=[[9415]])
    actual = _execution([[1]] * 100, truncated=True)

    assert compare_execution_result(actual, expected) is False


def test_truncated_case_counts_as_semantic_mismatch(demo_db: Path) -> None:
    case = EvaluationCase(
        id="truncated_semantic_mismatch",
        question="How many joined rows are there?",
        database_id="demo",
        expected_result=ExpectedResult(columns=["row_count"], rows=[[1000]]),
        fake_sql=(
            "SELECT c1.customer_id FROM customers AS c1 "
            "CROSS JOIN customers AS c2 CROSS JOIN customers AS c3"
        ),
    )
    report = Evaluator(demo_db, build_scripted_client).evaluate(
        [case], dataset_name="truncation", agent_mode="function-calling"
    )
    evaluated = report.cases[0]

    assert evaluated.protocol_completed is True
    assert evaluated.execution_success is True
    assert evaluated.answer_grounded is True
    assert evaluated.actual_result is not None
    assert evaluated.actual_result.truncated is True
    assert evaluated.result_correct is False
    assert evaluated.failure_kind is FailureKind.SEMANTIC_MISMATCH
    accuracy = report.mode_metrics["function-calling"].result_accuracy
    assert (accuracy.numerator, accuracy.denominator, accuracy.value) == (0, 1, 0)

    legacy_case = evaluated.model_copy(
        update={"result_correct": None, "failure_kind": None}
    )
    legacy_report = report.model_copy(update={"cases": [legacy_case]})
    rescored = rescore_evaluation_report(legacy_report)
    rescored_case = rescored.cases[0]
    rescored_accuracy = rescored.mode_metrics["function-calling"].result_accuracy
    assert rescored_case.result_correct is False
    assert rescored_case.failure_kind is FailureKind.SEMANTIC_MISMATCH
    assert (
        rescored_accuracy.numerator,
        rescored_accuracy.denominator,
        rescored_accuracy.value,
    ) == (0, 1, 0)


def test_unresolved_direct_answer_is_not_verifiable(demo_db: Path) -> None:
    dataset = load_behavior_dataset("data/synthetic/behavior_scenarios.json")
    case = next(case for case in dataset.cases if case.id == "direct_answer_ungrounded")

    def unresolved_client(_case: EvaluationCase, _mode: str) -> FakeLLMClient:
        return FakeLLMClient(
            [
                LLMResponse(
                    final_answer="direct answer",
                    response_mode=ResponseMode.PLAIN_FINAL,
                ),
                LLMResponse(
                    final_answer="still unresolved",
                    response_mode=ResponseMode.PLAIN_FINAL,
                ),
            ]
        )

    report = Evaluator(demo_db, unresolved_client).evaluate(
        [case],
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
    )
    evaluated = report.cases[0]
    assert evaluated.protocol_completed is False
    assert evaluated.execution_success is False
    assert evaluated.answer_grounded is False
    assert evaluated.result_correct is None
    assert evaluated.termination_reason == "unresolved_completion"
    assert evaluated.failure_kind is FailureKind.UNRESOLVED_COMPLETION
    assert report.mode_metrics["function-calling"].result_accuracy.denominator == 0
    assert report.mode_metrics["function-calling"].result_accuracy.value is None
    tool_metrics = report.mode_metrics["function-calling"].tool_metrics
    assert tool_metrics is not None
    assert tool_metrics.tool_sequence_exact_match_rate.value is None
    assert tool_metrics.required_tool_coverage.value is None
    assert tool_metrics.tool_execution_success_rate.value is None


def test_semantic_mismatch_is_evaluation_failure(demo_db: Path) -> None:
    dataset = load_behavior_dataset("data/synthetic/behavior_scenarios.json")
    case = next(case for case in dataset.cases if case.id == "semantic_mismatch")
    report = Evaluator(demo_db, build_scripted_client).evaluate(
        [case],
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
    )
    evaluated = report.cases[0]
    assert evaluated.execution_success is True
    assert evaluated.result_correct is False
    assert evaluated.failure_kind is FailureKind.SEMANTIC_MISMATCH
    metric = report.mode_metrics["function-calling"].semantic_mismatch_count
    assert (metric.numerator, metric.denominator, metric.value) == (1, 1, 1)


def test_one_case_exception_does_not_stop_later_cases(demo_db: Path) -> None:
    expected = ExpectedResult(columns=["customer_count"], rows=[[10]])
    cases = [
        EvaluationCase(
            id="broken",
            question="broken",
            database_id="demo",
            expected_result=expected,
        ),
        EvaluationCase(
            id="working",
            question="working",
            database_id="demo",
            expected_result=expected,
            fake_sql="SELECT COUNT(*) AS customer_count FROM customers",
        ),
    ]
    report = Evaluator(demo_db, build_scripted_client).evaluate(
        cases, dataset_name="isolation", agent_mode="pipeline", seed=1
    )
    by_id = {case.case_id: case for case in report.cases}
    assert by_id["broken"].failure_kind is FailureKind.UNRECOVERABLE_ERROR
    assert by_id["working"].result_correct is True


def test_expected_result_and_gold_sql_never_enter_initial_prompt(demo_db: Path) -> None:
    dataset = load_evaluation_dataset("data/synthetic/eval_questions.json")
    case = dataset.cases[0]
    clients: list[FakeLLMClient] = []

    def capture_client(selected: EvaluationCase, mode: str) -> FakeLLMClient:
        client = build_scripted_client(selected, mode)
        clients.append(client)
        return client

    Evaluator(demo_db, capture_client).evaluate(
        [case], dataset_name=dataset.dataset_name, agent_mode="both"
    )
    assert len(clients) == 2
    for client in clients:
        first_prompt = "\n".join(
            message.content or "" for message in client.requests[0].messages
        )
        assert "expected_result" not in first_prompt
        assert "gold_sql" not in first_prompt
        assert json.dumps(case.expected_result.rows, ensure_ascii=False) not in first_prompt


def test_pipeline_metrics_use_real_results(demo_db: Path) -> None:
    dataset = load_evaluation_dataset("data/synthetic/eval_questions.json")
    report = Evaluator(demo_db, build_scripted_client).evaluate(
        dataset.cases[:2],
        dataset_name=dataset.dataset_name,
        agent_mode="pipeline",
    )
    metrics = report.mode_metrics["pipeline"]
    assert (metrics.total_cases.numerator, metrics.total_cases.denominator) == (2, 1)
    assert metrics.final_execution_success_rate.value == 1
    assert metrics.result_accuracy.value == 1
    assert metrics.repair_success_rate.denominator == 0
    assert metrics.repair_success_rate.value is None


def test_function_tool_metrics_and_multiple_sequences(demo_db: Path) -> None:
    dataset = load_evaluation_dataset("data/synthetic/eval_questions.json")
    case = dataset.cases[0].model_copy(
        update={
            "expected_tool_sequences": [
                ["validate_sql", "execute_sql"],
                ["inspect_schema", "execute_sql"],
            ]
        }
    )
    report = Evaluator(demo_db, build_scripted_client).evaluate(
        [case],
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
    )
    tool = report.mode_metrics["function-calling"].tool_metrics
    assert tool is not None
    assert tool.tool_sequence_exact_match_rate.value == 1
    assert tool.required_tool_coverage.value == 1
    assert tool.valid_tool_argument_rate.value == 1
    assert tool.tool_execution_success_rate.value == 1
    assert tool.invalid_tool_call_rate.value == 0
    assert tool.average_tool_calls.value == 2


def test_required_tool_coverage_and_invalid_argument_rate(demo_db: Path) -> None:
    dataset = load_behavior_dataset("data/synthetic/behavior_scenarios.json")
    case = next(case for case in dataset.cases if case.id == "invalid_tool_arguments")
    report = Evaluator(demo_db, build_scripted_client).evaluate(
        [case],
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
    )
    tool = report.mode_metrics["function-calling"].tool_metrics
    assert tool is not None
    assert tool.required_tool_coverage.value == 1
    assert tool.valid_tool_argument_rate.value == 0.5
    assert tool.invalid_tool_call_rate.value == 0.5


def test_both_modes_run_identical_case_set(demo_db: Path) -> None:
    dataset = load_evaluation_dataset("data/synthetic/eval_questions.json")
    report = Evaluator(demo_db, build_scripted_client).evaluate(
        dataset.cases[:3],
        dataset_name=dataset.dataset_name,
        agent_mode=EvaluationAgentMode.BOTH,
        seed=42,
    )
    assert len(report.cases) == 6
    pipeline_ids = [case.case_id for case in report.cases if case.agent_mode == "pipeline"]
    function_ids = [
        case.case_id for case in report.cases if case.agent_mode == "function-calling"
    ]
    assert pipeline_ids == function_ids
    assert report.comparison is not None
    assert report.comparison.result_accuracy_difference == 0


def test_case_checkpoint_resume_skips_completed_cases_and_rejects_config_change(
    demo_db: Path, tmp_path: Path
) -> None:
    dataset = load_evaluation_dataset("data/synthetic/eval_questions.json")
    case = dataset.cases[0]
    checkpoint = tmp_path / "case_checkpoint.jsonl"
    fingerprint = "a" * 64
    first = Evaluator(demo_db, build_scripted_client).evaluate(
        [case],
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
        checkpoint_path=checkpoint,
        config_fingerprint=fingerprint,
    )

    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 1

    def unexpected_factory(_case: EvaluationCase, _mode: str) -> FakeLLMClient:
        raise AssertionError("completed checkpoint case was evaluated again")

    resumed = Evaluator(demo_db, unexpected_factory).evaluate(
        [case],
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
        checkpoint_path=checkpoint,
        resume=True,
        config_fingerprint=fingerprint,
    )

    assert resumed.cases == first.cases
    with pytest.raises(ValueError, match="fingerprint does not match"):
        Evaluator(demo_db, unexpected_factory).evaluate(
            [case],
            dataset_name=dataset.dataset_name,
            agent_mode="function-calling",
            checkpoint_path=checkpoint,
            resume=True,
            config_fingerprint="b" * 64,
        )


def test_behavior_scenarios_are_reproducible(demo_db: Path) -> None:
    dataset = load_behavior_dataset("data/synthetic/behavior_scenarios.json")
    first = Evaluator(demo_db, build_scripted_client).evaluate(
        dataset.cases,
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
        seed=7,
    )
    second = Evaluator(demo_db, build_scripted_client).evaluate(
        dataset.cases,
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
        seed=7,
    )
    first_facts = [
        (case.case_id, case.termination_reason, case.result_correct)
        for case in first.cases
    ]
    second_facts = [
        (case.case_id, case.termination_reason, case.result_correct)
        for case in second.cases
    ]
    assert first_facts == second_facts
    repaired = next(case for case in first.cases if case.case_id == "missing_table_repair")
    assert repaired.first_execution_success is False
    assert repaired.repair_succeeded is True
    assert repaired.result_correct is True


def test_three_reports_share_one_evaluation_report(
    demo_db: Path, tmp_path: Path
) -> None:
    dataset = load_evaluation_dataset("data/synthetic/eval_questions.json")
    report = Evaluator(demo_db, build_scripted_client).evaluate(
        dataset.cases[:2],
        dataset_name=dataset.dataset_name,
        agent_mode="both",
    )
    paths = write_reports(report, tmp_path)
    reloaded = EvaluationReport.model_validate_json(
        paths["json"].read_text(encoding="utf-8")
    )
    with paths["csv"].open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert reloaded == report
    assert len(csv_rows) == len(report.cases)
    assert str(report.mode_metrics["pipeline"].result_accuracy.numerator) in markdown
    assert "deterministic/mock" in markdown
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["cases"]
