"""Minimal run-only CLI tests for both agent modes."""

from pathlib import Path

import pytest

from execsql_agent.cli import _write_experiment_config, main


def test_pipeline_cli_runs_and_writes_trajectory(
    demo_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trajectory = tmp_path / "pipeline.jsonl"

    exit_code = main(
        [
            "run",
            "--agent-mode",
            "pipeline",
            "--database",
            str(demo_db),
            "--question",
            "消费金额最高的五名客户是谁？",
            "--trajectory-file",
            str(trajectory),
        ]
    )

    assert exit_code == 0
    assert trajectory.is_file()
    output = capsys.readouterr().out
    assert "终止原因：completed" in output
    assert "最终 SQL：" in output


def test_function_calling_cli_runs_and_writes_trajectory(
    demo_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trajectory = tmp_path / "function.jsonl"

    exit_code = main(
        [
            "run",
            "--agent-mode",
            "function-calling",
            "--database",
            str(demo_db),
            "--question",
            "消费金额最高的五名客户是谁？",
            "--trajectory-file",
            str(trajectory),
        ]
    )

    assert exit_code == 0
    assert trajectory.is_file()
    output = capsys.readouterr().out
    assert "终止原因：completed" in output
    assert "最终回答：" in output


def test_evaluate_cli_writes_three_reports(
    demo_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "reports"
    exit_code = main(
        [
            "evaluate",
            "--agent-mode",
            "both",
            "--database",
            str(demo_db),
            "--dataset",
            "data/synthetic/eval_questions.json",
            "--output-dir",
            str(output_dir),
            "--limit",
            "2",
            "--seed",
            "7",
        ]
    )
    assert exit_code == 0
    assert (output_dir / "evaluation_report.json").is_file()
    assert (output_dir / "case_results.csv").is_file()
    assert (output_dir / "evaluation_report.md").is_file()
    output = capsys.readouterr().out
    assert "评测完成：deterministic/mock" in output
    assert "JSON 报告：" in output


def test_experiment_config_resume_requires_identical_fingerprint(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bird-run"
    stable_config: dict[str, object] = {
        "model": "qwen3-8b-base",
        "dataset": {"revision": "frozen"},
        "temperature": 0.0,
    }

    fingerprint = _write_experiment_config(
        output_dir, stable_config, resume=False
    )
    assert _write_experiment_config(
        output_dir, stable_config, resume=True
    ) == fingerprint

    with pytest.raises(ValueError, match="fingerprint does not match"):
        _write_experiment_config(
            output_dir,
            {**stable_config, "model": "qwen3-8b-sft"},
            resume=True,
        )
