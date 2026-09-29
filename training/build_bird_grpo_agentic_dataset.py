"""Materialize the frozen BIRD GRPO pool for VERL ToolAgentLoop rollouts."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from execsql_agent.models import ExpectedResult, ToolCallRequest
from execsql_agent.rlvr.models import (
    DatabaseMetadata,
    ExtraInfo,
    GRPOSample,
    PromptMessage,
    VerifierMetadata,
)
from execsql_agent.tools.registry import ToolRegistry
from training.bird_phase_c import sha256_file
from training.build_bird_sft_dataset import _database_member, _extract_database
from training.verl_bird_agentic_adapter import (
    BIRD_GRPO_DATA_SOURCE,
    HARNESS_V2_SYSTEM_PROMPT,
    sample_to_agentic_record,
    write_agentic_parquet,
)
from training.verl_bird_agentic_adapter import (
    sha256_file as output_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data/bird/train/raw/train.json"
DEFAULT_POOL = PROJECT_ROOT / "data/bird/train/phase_c/bird_grpo_pool_v1.json"
DEFAULT_ELIGIBILITY = PROJECT_ROOT / "data/bird/train/phase_c/bird_execution_eligibility_v1.json"
DEFAULT_ARCHIVE = PROJECT_ROOT / "data/bird/train/raw/train_databases.zip"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/bird/train/phase_e15/bird_grpo_agentic_v1.parquet"
DEFAULT_DIAGNOSTIC_OUTPUT = (
    PROJECT_ROOT / "data/bird/train/phase_e15/bird_grpo_agentic_e2_diagnostic_v1.parquet"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/bird/train/phase_e15/bird_grpo_agentic_v1.manifest.json"
DEFAULT_TEMPORARY_ROOT = PROJECT_ROOT / "data/bird/train/tmp"
EXPECTED_CASES = 2500
EXPECTED_DATABASES = 61

EXPECTED_DIAGNOSTIC_CASES = 8


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _tool_schema_sha256(registry: ToolRegistry) -> str:
    payload = [
        {"type": "function", "function": item.model_dump(mode="json")}
        for item in registry.definitions
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _integrity_check(database: Path) -> None:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        result = connection.execute("PRAGMA quick_check").fetchone()
    if result != ("ok",):
        raise ValueError(f"SQLite quick_check failed: {database}: {result}")


def _expected_result(
    registry: ToolRegistry,
    *,
    case_id: str,
    sql: str,
) -> ExpectedResult:
    validation, result = registry.dispatch(
        ToolCallRequest(
            id=f"{case_id}_gold_execute",
            name="execute_sql",
            arguments={"sql": sql},
        )
    )
    execution = registry.execution_result(result)
    if (
        not validation.valid
        or not result.success
        or execution is None
        or not execution.execution_success
        or execution.truncated
    ):
        raise ValueError(f"{case_id}: frozen gold SQL is not execution-eligible")
    return ExpectedResult(
        columns=execution.columns,
        rows=execution.rows,
        ordered=False,
    )


def materialize(
    *,
    annotations_path: Path,
    pool_path: Path,
    eligibility_path: Path,
    database_archive_path: Path,
    output_path: Path,
    manifest_path: Path,
    diagnostic_output_path: Path,
    temporary_root: Path,
) -> dict[str, object]:
    """Build all 2500 records or leave no final Parquet on any hard failure."""

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    annotations = _load_json(annotations_path)
    if diagnostic_output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {diagnostic_output_path}")
    pool = _load_json(pool_path)
    eligibility = _load_json(eligibility_path)
    cases = pool.get("cases") if isinstance(pool, dict) else None
    eligible = eligibility.get("cases") if isinstance(eligibility, dict) else None
    if not isinstance(annotations, list) or not isinstance(cases, list):
        raise ValueError("invalid annotations or frozen GRPO pool")
    if not isinstance(eligible, dict):
        raise ValueError("invalid execution-eligibility manifest")
    if len(cases) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} pool cases, found {len(cases)}")
    database_ids = {str(case["database_id"]) for case in cases}
    if len(database_ids) != EXPECTED_DATABASES:
        raise ValueError(f"expected {EXPECTED_DATABASES} databases, found {len(database_ids)}")
    if any(
        case["case_id"] not in eligible or eligible[case["case_id"]].get("eligible") is not True
        for case in cases
    ):
        raise ValueError("frozen GRPO pool contains an execution-ineligible case")

    case_positions = {str(case["case_id"]): index for index, case in enumerate(cases)}

    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_database[str(case["database_id"])].append(case)
    temporary_root.mkdir(parents=True, exist_ok=True)

    indexed_records: dict[int, dict[str, object]] = {}
    database_counts: Counter[str] = Counter()
    database_hashes: dict[str, str] = {}
    tool_schema_sha256: str | None = None
    quick_check_count = 0
    failure: dict[str, object] | None = None

    with zipfile.ZipFile(database_archive_path) as archive:
        for db_index, db_id in enumerate(sorted(by_database), start=1):
            holder: tempfile.TemporaryDirectory[str] | None = None
            try:
                member = _database_member(archive, db_id)
                holder, database = _extract_database(
                    archive,
                    db_id=db_id,
                    temporary_root=temporary_root,
                )
                _integrity_check(database)
                quick_check_count += 1
                database_sha256 = sha256_file(database)
                database_hashes[db_id] = database_sha256
                registry = ToolRegistry(database)
                current_schema_hash = _tool_schema_sha256(registry)
                if tool_schema_sha256 is None:
                    tool_schema_sha256 = current_schema_hash
                elif current_schema_hash != tool_schema_sha256:
                    raise ValueError("ToolRegistry schema changed between databases")

                for pool_case in by_database[db_id]:
                    annotation_index = int(pool_case["annotation_index"])
                    row = annotations[annotation_index]
                    case_id = str(pool_case["case_id"])
                    if row.get("db_id") != db_id:
                        raise ValueError(f"{case_id}: annotation database mismatch")
                    question = row.get("question")
                    sql = row.get("SQL")
                    evidence = row.get("evidence")
                    if not isinstance(question, str) or not isinstance(sql, str):
                        raise ValueError(f"{case_id}: invalid annotation fields")
                    expected_result = _expected_result(
                        registry,
                        case_id=case_id,
                        sql=sql,
                    )
                    sample = GRPOSample(
                        sample_id=case_id,
                        data_source=BIRD_GRPO_DATA_SOURCE,
                        question=question,
                        messages=[
                            PromptMessage(
                                role="system",
                                content=HARNESS_V2_SYSTEM_PROMPT,
                            ),
                            PromptMessage(role="user", content=question),
                        ],
                        database_metadata=DatabaseMetadata(
                            database_id=db_id,
                            database_path=(
                                f"data/bird/train/runtime_databases/{db_id}/{db_id}.sqlite"
                            ),
                            database_sha256=database_sha256,
                            database_archive_path=str(
                                database_archive_path.resolve().relative_to(PROJECT_ROOT)
                            ),
                            database_archive_member=member,
                        ),
                        private_verifier_metadata=VerifierMetadata(
                            response_contract="agentic_tool_loop_v1",
                            expected_result=expected_result,
                        ),
                        extra_info=ExtraInfo(
                            dataset_name=BIRD_GRPO_DATA_SOURCE,
                            difficulty="unavailable",
                            evaluation_group="frozen_grpo_pool",
                            tags=["bird", "train", "grpo", "agentic"],
                            prompt_version="grpo_agentic_harness_v2",
                        ),
                    )
                    original_index = case_positions[case_id]
                    indexed_records[original_index] = sample_to_agentic_record(
                        sample,
                        index=original_index,
                        split="train",
                        gold_sql_for_scan=sql,
                        evidence_for_scan=(evidence if isinstance(evidence, str) else None),
                    )
                    database_counts[db_id] += 1
            except Exception as error:
                failure = {
                    "database_id": db_id,
                    "error_type": type(error).__name__,
                    "detail": str(error),
                }
                break
            finally:
                if holder is not None:
                    holder.cleanup()
            print(
                f"materialize {db_index}/{len(by_database)} {db_id}: "
                f"{len(by_database[db_id])} cases",
                flush=True,
            )

    audit: dict[str, object] = {
        "version": 1,
        "status": "failed" if failure else "ready",
        "contract": "VERL v0.9.0 native ToolAgentLoop + Harness-v2 ToolRegistry",
        "inputs": {
            "annotations": {
                "path": str(annotations_path),
                "sha256": sha256_file(annotations_path),
            },
            "grpo_pool": {
                "path": str(pool_path),
                "sha256": sha256_file(pool_path),
            },
            "execution_eligibility": {
                "path": str(eligibility_path),
                "sha256": sha256_file(eligibility_path),
            },
            "database_archive": {
                "path": str(database_archive_path),
                "size_bytes": database_archive_path.stat().st_size,
            },
        },
        "expected_cases": EXPECTED_CASES,
        "expected_databases": EXPECTED_DATABASES,
        "materialized_cases": len(indexed_records),
        "database_counts": dict(sorted(database_counts.items())),
        "database_sha256": dict(sorted(database_hashes.items())),
        "sqlite_quick_checks": quick_check_count,
        "tool_schema_sha256": tool_schema_sha256,
        "max_agent_steps": 6,
        "policy_visible": ["system", "question", "tool schemas", "tool observations"],
        "private_verifier_only": [
            "expected_result",
            "database_path",
            "database_sha256",
            "database_archive_path",
            "database_archive_member",
        ],
        "excluded_everywhere": ["gold_sql", "bird_evidence"],
        "failure": failure,
    }
    if failure or len(indexed_records) != EXPECTED_CASES:
        _atomic_json(manifest_path, audit)
        raise ValueError("agentic BIRD materializer hard-stop failed")

    records = [indexed_records[index] for index in range(EXPECTED_CASES)]
    sorted_database_ids = sorted(by_database)
    diagnostic_positions = [
        round(index * (len(sorted_database_ids) - 1) / (EXPECTED_DIAGNOSTIC_CASES - 1))
        for index in range(EXPECTED_DIAGNOSTIC_CASES)
    ]
    diagnostic_database_ids = [sorted_database_ids[index] for index in diagnostic_positions]
    diagnostic_case_ids = [
        str(by_database[database_id][0]["case_id"]) for database_id in diagnostic_database_ids
    ]
    diagnostic_records = [records[case_positions[case_id]] for case_id in diagnostic_case_ids]
    diagnostic_output = write_agentic_parquet(diagnostic_records, diagnostic_output_path)
    output = write_agentic_parquet(records, output_path)
    audit["output"] = {
        "path": str(output),
        "rows": len(records),
        "size_bytes": output.stat().st_size,
        "sha256": output_sha256(output),
    }
    audit["diagnostic_output"] = {
        "path": str(diagnostic_output),
        "rows": len(diagnostic_records),
        "database_ids": diagnostic_database_ids,
        "case_ids": diagnostic_case_ids,
        "size_bytes": diagnostic_output.stat().st_size,
        "sha256": output_sha256(diagnostic_output),
    }
    _atomic_json(manifest_path, audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the frozen BIRD GRPO pool for VERL ToolAgentLoop."
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--database-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_OUTPUT,
    )
    parser.add_argument(
        "--temporary-root",
        type=Path,
        default=DEFAULT_TEMPORARY_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = materialize(
        annotations_path=args.annotations.resolve(),
        pool_path=args.pool.resolve(),
        eligibility_path=args.eligibility.resolve(),
        database_archive_path=args.database_archive.resolve(),
        output_path=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
        temporary_root=args.temporary_root.resolve(),
        diagnostic_output_path=args.diagnostic_output.resolve(),
    )
    print(json.dumps(audit["output"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
