"""Rescore persisted evaluation reports without executing agents or calling an LLM."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from execsql_agent.evaluation.reports import write_reports
from execsql_agent.evaluation.rescore import rescore_evaluation_report
from execsql_agent.models import EvaluationReport


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rescore_directory(report_dir: Path) -> None:
    report_path = report_dir / "evaluation_report.json"
    trajectory_path = report_dir / "trajectories.jsonl"
    report = EvaluationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    before_hash = _sha256(trajectory_path)
    before_metrics = {
        mode: metrics.result_accuracy for mode, metrics in report.mode_metrics.items()
    }
    rescored = rescore_evaluation_report(report)
    write_reports(rescored, report_dir)
    after_hash = _sha256(trajectory_path)
    if before_hash != after_hash:
        raise RuntimeError(f"trajectory file changed unexpectedly: {trajectory_path}")
    for mode, metrics in rescored.mode_metrics.items():
        before = before_metrics[mode]
        after = metrics.result_accuracy
        print(
            f"{report_dir} [{mode}]: "
            f"{before.numerator:g}/{before.denominator} -> "
            f"{after.numerator:g}/{after.denominator}"
        )
    if before_hash is not None:
        print(f"trajectory sha256 unchanged: {before_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute reports from persisted case results without LLM calls."
    )
    parser.add_argument("report_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for report_dir in args.report_dirs:
        rescore_directory(report_dir)


if __name__ == "__main__":
    main()
