"""Agentic BIRD reward adapter for VERL v0.9.0 diagnostics."""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from execsql_agent.models import ExpectedResult
from execsql_agent.rlvr.verifier import SQLExecutionVerifier
from training.bird_phase_c import sha256_file
from training.build_bird_sft_dataset import _extract_database

RewardView = Literal["r1_exact", "r2_execution_shaped"]


def _failure(detail: str) -> dict[str, object]:
    return {
        "score": 0.0,
        "r1_score": 0.0,
        "r2_score": 0.0,
        "parse_status": "not_run",
        "validation_status": "not_run",
        "execution_status": "not_run",
        "comparison_status": "not_run",
        "failure_kind": "verifier_metadata_error",
        "detail": detail,
    }


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _private_metadata(value: object) -> Mapping[str, object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"ground_truth is not valid JSON: {error.msg}") from error
    private = _mapping(value, field="ground_truth")
    required = {"verifier_type", "response_contract", "expected_result"}
    if set(private) != required:
        raise ValueError(f"ground_truth must contain exactly {sorted(required)}")
    if private["verifier_type"] != "sqlite_execution_result_v1":
        raise ValueError("unsupported verifier_type")
    if private["response_contract"] != "agentic_tool_loop_v1":
        raise ValueError("unsupported response_contract")
    return private


def _database_metadata(extra_info: object) -> Mapping[str, object]:
    metadata = _mapping(extra_info, field="extra_info")
    database = _mapping(metadata.get("database"), field="extra_info.database")
    required = {
        "database_id",
        "database_path",
        "database_sha256",
        "database_archive_path",
        "database_archive_member",
    }
    if set(database) != required:
        raise ValueError(f"database metadata must contain exactly {sorted(required)}")
    return database


def _resolve_under_root(raw_path: object, root: Path, *, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{field} must be a non-empty string")
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field} escapes database_root")
    return resolved


def _scores(comparison_status: str) -> tuple[float, float]:
    if comparison_status == "matched":
        return 1.0, 1.0
    if comparison_status == "mismatched":
        return 0.0, 0.2
    return 0.0, 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: object,
    extra_info: Mapping[str, object] | None = None,
    *,
    database_root: str = ".",
    temporary_root: str | None = None,
    max_rows: int = 100,
    reward_view: RewardView = "r2_execution_shaped",
) -> dict[str, Any]:
    """Score the last agentic execute_sql call with deterministic SQLite execution."""

    del data_source
    if reward_view not in {"r1_exact", "r2_execution_shaped"}:
        return _failure(f"unsupported reward_view: {reward_view}")
    holder: tempfile.TemporaryDirectory[str] | None = None
    try:
        private = _private_metadata(ground_truth)
        database = _database_metadata(extra_info)
        root = Path(database_root).resolve()
        archive_path = _resolve_under_root(
            database["database_archive_path"],
            root,
            field="database_archive_path",
        )
        database_id = database["database_id"]
        member = database["database_archive_member"]
        expected_sha256 = database["database_sha256"]
        logical_path = database["database_path"]
        if not isinstance(database_id, str) or Path(database_id).name != database_id:
            raise ValueError("invalid database_id")
        expected_member = f"train_databases/{database_id}/{database_id}.sqlite"
        expected_logical = f"data/bird/train/runtime_databases/{database_id}/{database_id}.sqlite"
        if member != expected_member:
            raise ValueError("database_archive_member does not match database_id")
        if logical_path != expected_logical:
            raise ValueError("database_path does not match the trusted runtime layout")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("database_sha256 must be a 64-character string")
        if not archive_path.is_file():
            raise ValueError(f"trusted database archive does not exist: {archive_path}")

        temp_parent = (
            Path(temporary_root).resolve() if temporary_root else root / "data/bird/train/tmp"
        )
        temp_parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            holder, database_path = _extract_database(
                archive,
                db_id=database_id,
                temporary_root=temp_parent,
            )
        if sha256_file(database_path) != expected_sha256.casefold():
            raise ValueError("trusted database SHA-256 does not match metadata")
        expected = ExpectedResult.model_validate(private["expected_result"])
        verified = SQLExecutionVerifier(database_path, max_rows=max_rows).verify(
            solution_str,
            expected,
            response_contract="agentic_tool_loop_v1",
        )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        return _failure(str(error))
    finally:
        if holder is not None:
            holder.cleanup()

    r1_score, r2_score = _scores(verified.comparison_status.value)
    score = r1_score if reward_view == "r1_exact" else r2_score
    return {
        "score": score,
        "r1_score": r1_score,
        "r2_score": r2_score,
        "parse_status": verified.parse_status.value,
        "validation_status": verified.validation_status.value,
        "execution_status": verified.execution_status.value,
        "comparison_status": verified.comparison_status.value,
        "failure_kind": (
            verified.failure_kind.value if verified.failure_kind is not None else None
        ),
        "detail": verified.detail,
    }
