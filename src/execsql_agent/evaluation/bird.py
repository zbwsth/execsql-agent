"""Canonical BIRD Mini-Dev SQLite loading and scorer-side preflight."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter, sleep
from uuid import uuid4

from pydantic import Field, model_validator

from execsql_agent.models import (
    EvaluationCase,
    EvaluationDataset,
    ExpectedResult,
    StrictModel,
)
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator

BirdDatabaseResolver = Callable[[EvaluationCase], Path]
CACHE_VERSION = "1.0"


class _BirdRecord(StrictModel):
    """One canonical BIRD Mini-Dev SQLite record."""

    question_id: int | str
    db_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    sql: str = Field(alias="SQL", min_length=1)
    difficulty: str
    evidence: str


class BirdDatabaseFingerprint(StrictModel):
    """Content and filesystem identity for one canonical SQLite database."""

    size_bytes: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BirdGoldCacheEntry(StrictModel):
    """One independently reusable scorer-side gold result."""

    cache_version: str = CACHE_VERSION
    question_id: int | str
    case_id: str
    db_id: str
    difficulty: str
    gold_sql_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    database_fingerprint: BirdDatabaseFingerprint
    columns: list[str]
    rows: list[list[object]]
    row_count: int = Field(ge=0)
    gold_runtime_ms: float = Field(ge=0)
    generated_at: str

    @model_validator(mode="after")
    def validate_row_count(self) -> BirdGoldCacheEntry:
        """Reject incomplete or internally inconsistent cache records."""

        if self.row_count != len(self.rows):
            raise ValueError("BIRD gold cache row_count does not match rows")
        return self


class BirdCacheStats(StrictModel):
    """Cache decisions made during one complete adapter pass."""

    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    invalidated: int = Field(ge=0)


@dataclass(frozen=True)
class BirdPreflightResult:
    """Dataset plus cache facts needed to build a benchmark report."""

    dataset: EvaluationDataset
    cache_entries: list[BirdGoldCacheEntry]
    cache_stats: BirdCacheStats
    elapsed_ms: float


def resolve_bird_database_path(database_root: str | Path, db_id: str) -> Path:
    """Resolve exactly ``root/db_id/db_id.sqlite`` and reject path escapes."""

    root = Path(database_root).resolve()
    if not root.is_dir():
        raise ValueError(f"BIRD database root does not exist: {root}")
    if not db_id or Path(db_id).name != db_id or "/" in db_id or "\\" in db_id:
        raise ValueError(f"Invalid BIRD db_id: {db_id!r}")

    database_path = (root / db_id / f"{db_id}.sqlite").resolve()
    try:
        database_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"BIRD database path escapes root: {db_id!r}") from error
    if database_path.name != f"{db_id}.sqlite":
        raise ValueError(f"BIRD database filename does not match db_id: {db_id!r}")
    if not database_path.is_file():
        raise ValueError(f"BIRD database does not exist: {database_path}")
    return database_path


def make_bird_database_resolver(database_root: str | Path) -> BirdDatabaseResolver:
    """Build a case resolver with a fixed, validated database root."""

    root = Path(database_root).resolve()
    if not root.is_dir():
        raise ValueError(f"BIRD database root does not exist: {root}")

    def resolve(case: EvaluationCase) -> Path:
        return resolve_bird_database_path(root, case.database_id)

    return resolve


def _read_records(path: Path) -> list[object]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"BIRD dataset is empty: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records: list[object] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid BIRD JSONL at line {line_number}: {error.msg}"
                ) from error
        return records
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("BIRD dataset must be a JSON array or JSONL objects.")


def _hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _database_fingerprint(path: Path) -> BirdDatabaseFingerprint:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return BirdDatabaseFingerprint(
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=digest.hexdigest(),
    )


def _read_gold_cache(path: Path) -> dict[str, BirdGoldCacheEntry]:
    if not path.exists():
        return {}
    entries: dict[str, BirdGoldCacheEntry] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entry = BirdGoldCacheEntry.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"Invalid BIRD gold cache at line {line_number}: {error}"
                ) from error
            if entry.case_id in entries:
                raise ValueError(f"Duplicate BIRD gold cache case: {entry.case_id}")
            entries[entry.case_id] = entry
    return entries


def _write_gold_cache_atomic(
    path: Path, entries: dict[str, BirdGoldCacheEntry]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for case_id in sorted(entries):
                stream.write(entries[case_id].model_dump_json())
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                sleep(0.05 * (2**attempt))
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:
                pass


def _cache_entry_is_valid(
    entry: BirdGoldCacheEntry,
    record: _BirdRecord,
    case_id: str,
    sql_hash: str,
    database_fingerprint: BirdDatabaseFingerprint,
) -> bool:
    return (
        entry.cache_version == CACHE_VERSION
        and entry.question_id == record.question_id
        and entry.case_id == case_id
        and entry.db_id == record.db_id
        and entry.difficulty == record.difficulty
        and entry.gold_sql_sha256 == sql_hash
        and entry.database_fingerprint == database_fingerprint
    )


def _validate_records(
    path: str | Path, expected_count: int
) -> tuple[Path, list[object]]:
    if expected_count < 1:
        raise ValueError("expected_count must be at least 1")
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise ValueError(f"BIRD dataset does not exist: {dataset_path}")
    raw_records = _read_records(dataset_path)
    if len(raw_records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} BIRD cases, found {len(raw_records)}."
        )
    return dataset_path, raw_records


def _load_bird_dataset(
    path: str | Path,
    database_root: str | Path,
    *,
    expected_count: int,
    gold_cache_path: Path | None,
    gold_cache_read_only: bool,
    gold_timeout_seconds: float,
    on_case: Callable[[BirdGoldCacheEntry, bool, int, int], None] | None,
) -> BirdPreflightResult:
    if gold_timeout_seconds <= 0:
        raise ValueError("gold_timeout_seconds must be positive")
    started_at = perf_counter()
    _, raw_records = _validate_records(path, expected_count)
    cached_entries = (
        _read_gold_cache(gold_cache_path) if gold_cache_path is not None else {}
    )
    validator = SQLValidator()
    cases: list[EvaluationCase] = []
    used_entries: list[BirdGoldCacheEntry] = []
    seen_ids: set[str] = set()
    fingerprints: dict[Path, BirdDatabaseFingerprint] = {}
    allowed_difficulties = {"simple", "moderate", "challenging"}
    hits = 0
    misses = 0
    invalidated = 0

    for index, raw_record in enumerate(raw_records, start=1):
        record = _BirdRecord.model_validate(raw_record)
        if record.difficulty not in allowed_difficulties:
            raise ValueError(
                f"Invalid BIRD difficulty for question {record.question_id!r}: "
                f"{record.difficulty!r}"
            )
        case_id = f"bird_{record.question_id}"
        if case_id in seen_ids:
            raise ValueError(f"Duplicate BIRD question_id: {record.question_id!r}")
        seen_ids.add(case_id)

        database_path = resolve_bird_database_path(database_root, record.db_id)
        safety = validator.validate(record.sql, database_path=database_path)
        if not safety.safe:
            raise ValueError(
                f"BIRD case {case_id} is not a read-only SELECT/WITH query: "
                f"{safety.reason}"
            )
        if safety.syntax_valid is not True:
            raise ValueError(
                f"BIRD gold SQL failed validation for {case_id}: "
                f"{safety.validation_error}"
            )

        database_fingerprint = fingerprints.get(database_path)
        if database_fingerprint is None:
            database_fingerprint = _database_fingerprint(database_path)
            fingerprints[database_path] = database_fingerprint
        sql_hash = _hash_text(record.sql)
        cached = cached_entries.get(case_id)
        cache_hit = cached is not None and _cache_entry_is_valid(
            cached,
            record,
            case_id,
            sql_hash,
            database_fingerprint,
        )
        if cache_hit:
            entry = cached
            if entry is None:  # pragma: no cover - narrows the validated branch
                raise RuntimeError("validated cache entry disappeared")
            hits += 1
        else:
            misses += 1
            if cached is not None:
                invalidated += 1
            if gold_cache_read_only:
                cache_state = "invalid" if cached is not None else "missing"
                raise ValueError(
                    f"BIRD gold cache entry is {cache_state} for {case_id}; "
                    "frozen evaluation will not execute gold SQL or modify the cache."
                )
            gold_execution = SQLExecutor(
                database_path, validator=validator
            ).execute_full(record.sql, timeout_seconds=gold_timeout_seconds)
            if not gold_execution.execution_success:
                error_message = (
                    gold_execution.error.message
                    if gold_execution.error is not None
                    else gold_execution.blocked_reason or "unknown execution failure"
                )
                raise ValueError(
                    "BIRD gold SQL failed full execution: "
                    f"question_id={record.question_id!r}, db_id={record.db_id!r}, "
                    f"difficulty={record.difficulty!r}, "
                    f"elapsed_ms={gold_execution.duration_ms:.3f}, "
                    f"error={error_message}, SQL={record.sql}"
                )
            if gold_execution.truncated:
                raise ValueError(f"BIRD gold SQL was unexpectedly truncated: {case_id}")
            entry = BirdGoldCacheEntry(
                question_id=record.question_id,
                case_id=case_id,
                db_id=record.db_id,
                difficulty=record.difficulty,
                gold_sql_sha256=sql_hash,
                database_fingerprint=database_fingerprint,
                columns=gold_execution.columns,
                rows=gold_execution.rows,
                row_count=gold_execution.returned_row_count,
                gold_runtime_ms=gold_execution.duration_ms,
                generated_at=datetime.now(UTC).isoformat(),
            )
            if gold_cache_path is not None:
                cached_entries[case_id] = entry
                _write_gold_cache_atomic(gold_cache_path, cached_entries)

        used_entries.append(entry)
        expected_result = ExpectedResult(
            columns=entry.columns,
            rows=entry.rows,
            ordered=False,
        )
        cases.append(
            EvaluationCase(
                id=case_id,
                question=record.question,
                database_id=record.db_id,
                expected_result=expected_result,
                gold_sql=record.sql,
                gold_runtime_ms=entry.gold_runtime_ms,
                difficulty=record.difficulty,
                evidence=record.evidence or None,
                comparison_mode="bird_set",
                tags=["bird", record.difficulty],
            )
        )
        if on_case is not None:
            on_case(entry, cache_hit, index, len(raw_records))

    return BirdPreflightResult(
        dataset=EvaluationDataset(dataset_name="bird_mini_dev_sqlite", cases=cases),
        cache_entries=used_entries,
        cache_stats=BirdCacheStats(
            hits=hits,
            misses=misses,
            invalidated=invalidated,
        ),
        elapsed_ms=(perf_counter() - started_at) * 1000,
    )


def load_bird_dataset(
    path: str | Path,
    database_root: str | Path,
    *,
    expected_count: int = 500,
    gold_cache_path: str | Path | None = None,
    gold_cache_read_only: bool = False,
    gold_timeout_seconds: float = 900.0,
) -> EvaluationDataset:
    """Load, validate, and preflight canonical BIRD Mini-Dev SQLite cases."""

    return _load_bird_dataset(
        path,
        database_root,
        expected_count=expected_count,
        gold_cache_path=(
            Path(gold_cache_path) if gold_cache_path is not None else None
        ),
        gold_cache_read_only=gold_cache_read_only,
        gold_timeout_seconds=gold_timeout_seconds,
        on_case=None,
    ).dataset


def preflight_bird_dataset(
    path: str | Path,
    database_root: str | Path,
    gold_cache_path: str | Path,
    *,
    expected_count: int = 500,
    gold_timeout_seconds: float = 900.0,
    on_case: Callable[[BirdGoldCacheEntry, bool, int, int], None] | None = None,
) -> BirdPreflightResult:
    """Build a resumable, fingerprint-validated BIRD gold result cache."""

    return _load_bird_dataset(
        path,
        database_root,
        expected_count=expected_count,
        gold_cache_path=Path(gold_cache_path),
        gold_cache_read_only=False,
        gold_timeout_seconds=gold_timeout_seconds,
        on_case=on_case,
    )
