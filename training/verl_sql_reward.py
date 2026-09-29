"""verl v0.9.0 custom reward adapter for ExecSQL-Agent SQL RLVR."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from execsql_agent.models import ExpectedResult
from execsql_agent.rlvr.verifier import SQLExecutionVerifier


def _failure(detail: str) -> dict[str, object]:
    return {
        "score": 0.0,
        "parse_status": "not_run",
        "validation_status": "not_run",
        "execution_status": "not_run",
        "comparison_status": "not_run",
        "failure_kind": "verifier_metadata_error",
        "detail": detail,
    }


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _parse_ground_truth(value: object) -> Mapping[str, object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"ground_truth is not valid JSON: {error.msg}") from error
    ground_truth = _as_mapping(value, field="ground_truth")
    if set(ground_truth) != {"verifier_type", "expected_result"}:
        raise ValueError("ground_truth must contain exactly verifier_type and expected_result")
    if ground_truth.get("verifier_type") != "sqlite_execution_result_v1":
        raise ValueError("unsupported verifier_type")
    return ground_truth


def _resolve_database(
    extra_info: Mapping[str, object], database_root: str | None
) -> tuple[Path, str]:
    database = _as_mapping(extra_info.get("database"), field="extra_info.database")
    if set(database) != {"database_id", "database_path", "database_sha256"}:
        raise ValueError("database metadata has an unexpected shape")
    raw_path = database.get("database_path")
    expected_sha256 = database.get("database_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("database_path must be a non-empty trusted metadata string")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("database_sha256 must be a 64-character trusted metadata value")

    root = Path(database_root).resolve() if database_root is not None else Path.cwd().resolve()
    configured = Path(raw_path)
    resolved = configured.resolve() if configured.is_absolute() else (root / configured).resolve()
    if database_root is not None and not resolved.is_relative_to(root):
        raise ValueError("trusted database metadata escapes database_root")
    return resolved, expected_sha256.casefold()


@lru_cache(maxsize=32)
def _database_sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: object,
    extra_info: Mapping[str, object] | None = None,
    *,
    database_root: str | None = None,
    max_rows: int = 100,
) -> dict[str, Any]:
    """Adapt verl's v0.9.0 reward callback arguments to SQLExecutionVerifier."""

    del data_source
    try:
        private = _parse_ground_truth(ground_truth)
        metadata = _as_mapping(extra_info, field="extra_info")
        database_path, expected_sha256 = _resolve_database(metadata, database_root)
        if not database_path.is_file():
            return _failure(f"Trusted database does not exist: {database_path}")
        stat = database_path.stat()
        actual_sha256 = _database_sha256(str(database_path), stat.st_size, stat.st_mtime_ns)
        if actual_sha256 != expected_sha256:
            return _failure("Trusted database SHA-256 does not match dataset metadata")
        expected_result = ExpectedResult.model_validate(private["expected_result"])
    except (OSError, ValueError) as error:
        return _failure(str(error))

    verified = SQLExecutionVerifier(database_path, max_rows=max_rows).verify(
        solution_str,
        expected_result,
    )
    return {
        "score": verified.reward,
        "parse_status": verified.parse_status.value,
        "validation_status": verified.validation_status.value,
        "execution_status": verified.execution_status.value,
        "comparison_status": verified.comparison_status.value,
        "failure_kind": (
            verified.failure_kind.value if verified.failure_kind is not None else None
        ),
        "detail": verified.detail,
    }
