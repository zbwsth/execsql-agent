"""Render JSON, CSV, and Markdown from one immutable EvaluationReport."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

from execsql_agent.models import EvaluationReport, MetricValue, ModeMetrics


def _metric_items(metrics: ModeMetrics) -> Iterator[tuple[str, MetricValue]]:
    yield "total_cases", metrics.total_cases
    yield "execution_accuracy", metrics.execution_accuracy
    yield "protocol_completion_rate", metrics.protocol_completion_rate
    yield "first_execution_success_rate", metrics.first_execution_success_rate
    yield "final_execution_success_rate", metrics.final_execution_success_rate
    yield "result_accuracy", metrics.result_accuracy
    yield "grounded_answer_rate", metrics.grounded_answer_rate
    yield "repair_success_rate", metrics.repair_success_rate
    yield "refusal_accuracy", metrics.refusal_accuracy
    yield "unsafe_sql_block_rate", metrics.unsafe_sql_block_rate
    yield "semantic_mismatch_count", metrics.semantic_mismatch_count
    yield "unsafe_sql_rate", metrics.unsafe_sql_rate
    yield "repeated_sql_rate", metrics.repeated_sql_rate
    yield "average_llm_turns", metrics.average_llm_turns
    yield "average_sql_executions", metrics.average_sql_executions
    yield "average_duration_ms", metrics.average_duration_ms
    if metrics.tool_metrics is not None:
        tool = metrics.tool_metrics
        yield "tool.tool_sequence_exact_match_rate", tool.tool_sequence_exact_match_rate
        yield "tool.required_tool_coverage", tool.required_tool_coverage
        yield "tool.valid_tool_argument_rate", tool.valid_tool_argument_rate
        yield "tool.tool_execution_success_rate", tool.tool_execution_success_rate
        yield "tool.invalid_tool_call_rate", tool.invalid_tool_call_rate
        yield "tool.repeated_tool_call_rate", tool.repeated_tool_call_rate
        yield "tool.average_tool_calls", tool.average_tool_calls
    if metrics.prediction_sql_runtime is not None:
        yield "prediction_sql.timeout_rate", metrics.prediction_sql_runtime.timeout_rate


def _format_value(value: float | None) -> str:
    return "null" if value is None else f"{value:.6f}"


def render_markdown(report: EvaluationReport) -> str:
    """Render a human-readable summary without recalculating any metric."""

    lines = [
        "# Agent 评测报告",
        "",
        f"- 数据集：`{report.dataset_name}`",
        f"- 数据库：`{report.database_id}`",
        f"- 数据库集合：`{', '.join(report.database_ids)}`",
        f"- Agent 模式：`{report.agent_mode.value}`",
        f"- 运行模式：`{report.run_mode}`",
        f"- Oracle evidence：`{str(report.oracle_evidence).lower()}`",
        f"- 随机种子：`{report.seed}`",
    ]
    if report.run_mode == "deterministic/mock":
        lines.append("- 说明：FakeLLM 结果仅表示流程可复现，不代表真实模型泛化能力。")
    else:
        lines.append("- 说明：指标来自本次真实模型调用及真实 SQLite 执行。")
    lines.append("")
    for mode, metrics in report.mode_metrics.items():
        lines.extend(
            [
                f"## {mode}",
                "",
                "| 指标 | numerator | denominator | value |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, metric_value in _metric_items(metrics):
            lines.append(
                f"| `{name}` | {metric_value.numerator:g} | "
                f"{metric_value.denominator} | {_format_value(metric_value.value)} |"
            )
        lines.append("")
        if metrics.stable_bird_ex is not None:
            lines.extend(
                [
                    "### Stable BIRD EX",
                    "",
                    "| difficulty | numerator | denominator | value |",
                    "|---|---:|---:|---:|",
                ]
            )
            for difficulty, metric_value in metrics.stable_bird_ex.model_dump().items():
                metric = MetricValue.model_validate(metric_value)
                lines.append(
                    f"| `{difficulty}` | {metric.numerator:g} | "
                    f"{metric.denominator} | {_format_value(metric.value)} |"
                )
            lines.append("")
        if metrics.ex_at_30s is not None:
            lines.extend(
                [
                    "### EX@30s",
                    "",
                    "| difficulty | numerator | denominator | value |",
                    "|---|---:|---:|---:|",
                ]
            )
            for difficulty, metric_value in metrics.ex_at_30s.model_dump().items():
                metric = MetricValue.model_validate(metric_value)
                lines.append(
                    f"| `{difficulty}` | {metric.numerator:g} | "
                    f"{metric.denominator} | {_format_value(metric.value)} |"
                )
            lines.append("")
        if metrics.prediction_sql_runtime is not None:
            runtime = metrics.prediction_sql_runtime
            lines.extend(
                [
                    "### Prediction SQL Runtime",
                    "",
                    f"- measured count: `{runtime.measured_count}`",
                    f"- median ms: `{runtime.median_ms}`",
                    f"- p95 ms: `{runtime.p95_ms}`",
                    f"- timeout count: `{runtime.timeout_count}`",
                    "",
                ]
            )

    if report.comparison is not None:
        lines.extend(["## 模式差值", "", "Function Calling - Pipeline：", ""])
        for name, value in report.comparison.model_dump().items():
            lines.append(f"- `{name}`：{_format_value(value)}")
        lines.append("")

    failures = [case for case in report.cases if case.failure_kind is not None]
    lines.extend(["## 失败案例", ""])
    if not failures:
        lines.append("无。")
    else:
        lines.extend(
            [
                "| case_id | mode | failure_kind | termination_reason |",
                "|---|---|---|---|",
            ]
        )
        for case in failures:
            failure_kind = case.failure_kind.value if case.failure_kind else ""
            lines.append(
                f"| `{case.case_id}` | `{case.agent_mode}` | `{failure_kind}` | "
                f"`{case.termination_reason}` |"
            )
    lines.append("")
    return "\n".join(lines)


def write_reports(report: EvaluationReport, output_dir: str | Path) -> dict[str, Path]:
    """Write three views of the same validated in-memory report."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": target / "evaluation_report.json",
        "csv": target / "case_results.csv",
        "markdown": target / "evaluation_report.md",
    }
    paths["json"].write_text(
        report.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )
    fieldnames = [
        "case_id",
        "dataset_name",
        "database_id",
        "agent_mode",
        "difficulty",
        "comparison_mode",
        "gold_runtime_ms",
        "prediction_timeout_budget_seconds",
        "prediction_runtime_ms",
        "prediction_timeout",
        "stable_bird_ex",
        "ex_at_30s",
        "tags",
        "protocol_completed",
        "execution_success",
        "answer_grounded",
        "result_correct",
        "failure_kind",
        "termination_reason",
        "first_execution_success",
        "repair_succeeded",
        "total_llm_turns",
        "total_tool_calls",
        "total_sql_executions",
        "invalid_tool_calls",
        "repeated_tool_calls",
        "unsafe_sql_count",
        "total_duration_ms",
        "final_sql",
        "final_answer",
        "actual_tool_sequence",
        "error_message",
    ]
    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for case in report.cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "dataset_name": case.dataset_name,
                    "database_id": case.database_id,
                    "agent_mode": case.agent_mode,
                    "difficulty": case.difficulty or "",
                    "comparison_mode": case.comparison_mode,
                    "gold_runtime_ms": case.gold_runtime_ms,
                    "prediction_timeout_budget_seconds": (
                        case.prediction_timeout_budget_seconds
                    ),
                    "prediction_runtime_ms": case.prediction_runtime_ms,
                    "prediction_timeout": case.prediction_timeout,
                    "stable_bird_ex": case.stable_bird_ex,
                    "ex_at_30s": case.ex_at_30s,
                    "tags": json.dumps(case.tags, ensure_ascii=False),
                    "protocol_completed": case.protocol_completed,
                    "execution_success": case.execution_success,
                    "answer_grounded": case.answer_grounded,
                    "result_correct": case.result_correct,
                    "failure_kind": (
                        case.failure_kind.value if case.failure_kind else ""
                    ),
                    "termination_reason": case.termination_reason,
                    "first_execution_success": case.first_execution_success,
                    "repair_succeeded": case.repair_succeeded,
                    "total_llm_turns": case.total_llm_turns,
                    "total_tool_calls": case.total_tool_calls,
                    "total_sql_executions": case.total_sql_executions,
                    "invalid_tool_calls": case.invalid_tool_calls,
                    "repeated_tool_calls": case.repeated_tool_calls,
                    "unsafe_sql_count": case.unsafe_sql_count,
                    "total_duration_ms": case.total_duration_ms,
                    "final_sql": case.final_sql or "",
                    "final_answer": case.final_answer or "",
                    "actual_tool_sequence": json.dumps(
                        case.actual_tool_sequence, ensure_ascii=False
                    ),
                    "error_message": case.error_message or "",
                }
            )
    paths["markdown"].write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    return {name: path.resolve() for name, path in paths.items()}
