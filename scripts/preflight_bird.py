"""Build and audit the canonical BIRD Mini-Dev scorer-side gold cache."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import median

from execsql_agent.evaluation.bird import (
    CACHE_VERSION,
    BirdGoldCacheEntry,
    preflight_bird_dataset,
)
from execsql_agent.evaluation.comparator import compare_bird_execution_result
from execsql_agent.models import ExecutionResult, ExpectedResult

DATASET_REVISION = "f65faf4ae3b638c1fa6df1d3370c8d92c8366301"
DATASET_SOURCE = "https://huggingface.co/datasets/birdsql/bird_mini_dev"
DATASET_SHA256 = "88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2"
DATABASE_PACKAGE_URL = (
    "https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view"
)
DATABASE_PACKAGE_SHA256 = (
    "aeb211c0e39010bbdae3838bb5e8bd27dc446ed77495b1709f85ccc9bf67f2be"
)


def _percentile_95(values: list[float] | list[int]) -> float | int:
    ordered = sorted(values)
    return ordered[ceil(0.95 * len(ordered)) - 1]


def _runtime_bucket(runtime_ms: float) -> str:
    seconds = runtime_ms / 1000
    if seconds < 1:
        return "lt_1s"
    if seconds < 5:
        return "1_to_lt_5s"
    if seconds < 30:
        return "5_to_lt_30s"
    if seconds < 60:
        return "30_to_lt_60s"
    if seconds <= 300:
        return "60_to_300s"
    return "gt_300s"


def _smoke(entries: list[BirdGoldCacheEntry]) -> dict[str, object]:
    correct_by_difficulty: Counter[str] = Counter()
    total_by_difficulty: Counter[str] = Counter()
    overall_correct = 0
    for entry in entries:
        expected = ExpectedResult(
            columns=entry.columns,
            rows=entry.rows,
            ordered=False,
        )
        actual = ExecutionResult(
            executed=True,
            execution_success=True,
            columns=entry.columns,
            rows=entry.rows,
            returned_row_count=entry.row_count,
            truncated=False,
            duration_ms=entry.gold_runtime_ms,
        )
        correct = compare_bird_execution_result(actual, expected)
        overall_correct += correct
        total_by_difficulty[entry.difficulty] += 1
        correct_by_difficulty[entry.difficulty] += correct
    return {
        "overall": {
            "correct": overall_correct,
            "denominator": len(entries),
            "ex": overall_correct / len(entries),
        },
        "difficulty": {
            difficulty: {
                "correct": correct_by_difficulty[difficulty],
                "denominator": total_by_difficulty[difficulty],
                "ex": (
                    correct_by_difficulty[difficulty]
                    / total_by_difficulty[difficulty]
                ),
            }
            for difficulty in ("simple", "moderate", "challenging")
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_report(
    entries: list[BirdGoldCacheEntry],
    *,
    elapsed_ms: float,
    cache_hits: int,
    cache_misses: int,
    cache_invalidated: int,
    cache_path: Path,
) -> dict[str, object]:
    runtimes = [entry.gold_runtime_ms for entry in entries]
    row_counts = [entry.row_count for entry in entries]
    difficulty = Counter(entry.difficulty for entry in entries)
    per_database = Counter(entry.db_id for entry in entries)
    runtime_buckets = Counter(_runtime_bucket(runtime) for runtime in runtimes)
    slow_cases = [
        {
            "question_id": entry.question_id,
            "case_id": entry.case_id,
            "db_id": entry.db_id,
            "difficulty": entry.difficulty,
            "runtime_ms": entry.gold_runtime_ms,
        }
        for entry in entries
        if entry.gold_runtime_ms > 30_000
    ]
    over_100 = [
        {
            "question_id": entry.question_id,
            "case_id": entry.case_id,
            "db_id": entry.db_id,
            "difficulty": entry.difficulty,
            "row_count": entry.row_count,
        }
        for entry in entries
        if entry.row_count > 100
    ]
    by_question = {str(entry.question_id): entry for entry in entries}
    return {
        "dataset_name": "bird_mini_dev_sqlite",
        "dataset_source": DATASET_SOURCE,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA256,
        "database_package_provenance": {
            "url": DATABASE_PACKAGE_URL,
            "filename": "minidev_0703.zip",
            "sha256": DATABASE_PACKAGE_SHA256,
        },
        "case_count": len(entries),
        "unique_case_count": len({entry.case_id for entry in entries}),
        "database_count": len(per_database),
        "difficulty": {
            name: difficulty[name]
            for name in ("simple", "moderate", "challenging")
        },
        "per_database": [
            {"db_id": db_id, "case_count": count}
            for db_id, count in sorted(per_database.items())
        ],
        "gold_execution": {
            "success_count": len(entries),
            "failure_count": 0,
            "preflight_elapsed_ms": elapsed_ms,
        },
        "gold_runtime": {
            "buckets": {
                name: runtime_buckets[name]
                for name in (
                    "lt_1s",
                    "1_to_lt_5s",
                    "5_to_lt_30s",
                    "30_to_lt_60s",
                    "60_to_300s",
                    "gt_300s",
                )
            },
            "min_ms": min(runtimes),
            "median_ms": median(runtimes),
            "p95_ms": _percentile_95(runtimes),
            "max_ms": max(runtimes),
            "cases_over_30s": len(slow_cases),
        },
        "slow_gold_cases": slow_cases,
        "special_gold_cases": {
            question_id: {
                "db_id": by_question[question_id].db_id,
                "difficulty": by_question[question_id].difficulty,
                "runtime_ms": by_question[question_id].gold_runtime_ms,
                "row_count": by_question[question_id].row_count,
            }
            for question_id in ("701", "518")
        },
        "result_size": {
            "min_rows": min(row_counts),
            "median_rows": median(row_counts),
            "p95_rows": _percentile_95(row_counts),
            "max_rows": max(row_counts),
            "cases_over_100_rows": len(over_100),
        },
        "over_100_row_cases": over_100,
        "cache": {
            "version": CACHE_VERSION,
            "path": cache_path.as_posix(),
            "hits": cache_hits,
            "misses": cache_misses,
            "invalidated": cache_invalidated,
        },
        "gold_as_reference_smoke": _smoke(entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/bird/mini_dev_sqlite.json"),
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path("data/bird/dev_databases"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/bird/cache/mini_dev_gold_results.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/bird/preflight_report.json"),
    )
    parser.add_argument("--gold-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    def progress(
        entry: BirdGoldCacheEntry, cache_hit: bool, index: int, total: int
    ) -> None:
        status = "hit" if cache_hit else "miss"
        print(
            f"[{index}/{total}] {entry.case_id} db={entry.db_id} "
            f"cache={status} runtime_ms={entry.gold_runtime_ms:.3f} "
            f"rows={entry.row_count}",
            flush=True,
        )

    result = preflight_bird_dataset(
        args.dataset,
        args.database_root,
        args.cache,
        gold_timeout_seconds=args.gold_timeout_seconds,
        on_case=progress,
    )
    report = _build_report(
        result.cache_entries,
        elapsed_ms=result.elapsed_ms,
        cache_hits=result.cache_stats.hits,
        cache_misses=result.cache_stats.misses,
        cache_invalidated=result.cache_stats.invalidated,
        cache_path=args.cache,
    )
    _write_json_atomic(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
