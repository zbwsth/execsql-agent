"""Thin multi-database BIRD SFT screening and orchestration."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from execsql_agent.models import ToolCallRequest, ToolCallResult
from execsql_agent.tools.registry import ToolRegistry
from training.assistant_turn_preprocessing import process_trajectory
from training.bird_phase_c import analyze_sql, sha256_file
from training.build_sft_dataset import build_dataset

DEFAULT_MAX_LENGTH = 4096
DEFAULT_SCHEMA_OBSERVATION_CHAR_LIMIT = 12_000


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dispatch(
    registry: ToolRegistry,
    *,
    call_id: str,
    name: str,
    arguments: dict[str, object],
) -> ToolCallResult:
    validation, result = registry.dispatch(
        ToolCallRequest(id=call_id, name=name, arguments=arguments)
    )
    if not validation.valid:
        raise ValueError(
            f"{call_id}: invalid {name}: "
            f"{validation.error_code}: {validation.error_message}"
        )
    if not result.success:
        raise ValueError(
            f"{call_id}: failed {name}: "
            f"{result.error_code}: {result.error_message}"
        )
    return result


def _listed_tables(result: ToolCallResult) -> list[str]:
    output = result.output
    if not isinstance(output, dict):
        raise ValueError("list_tables result has no output object")
    raw = output.get("tables")
    if not isinstance(raw, list) or not all(
        isinstance(table, str) for table in raw
    ):
        raise ValueError("list_tables result has no tables array")
    return raw


def _runtime_physical_tables(database: Path, sql: str) -> set[str]:
    seen: set[str] = set()

    def authorizer(
        action: int,
        argument_1: str | None,
        _argument_2: str | None,
        _database: str | None,
        _trigger: str | None,
    ) -> int:
        if action == sqlite3.SQLITE_READ and argument_1:
            if not argument_1.casefold().startswith("sqlite_"):
                seen.add(argument_1)
        return sqlite3.SQLITE_OK

    uri = f"{database.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(authorizer)
        connection.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    return seen


def _database_member(archive: zipfile.ZipFile, db_id: str) -> str:
    expected = f"train_databases/{db_id}/{db_id}.sqlite"
    if expected not in set(archive.namelist()):
        raise FileNotFoundError(
            f"Database member not found in archive: {expected}"
        )
    return expected


def _drop_file_cache(file_descriptor: int, length: int) -> None:
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advice is None or not hasattr(os, "posix_fadvise"):
        return
    os.posix_fadvise(file_descriptor, 0, length, advice)


def _extract_database(
    archive: zipfile.ZipFile,
    *,
    db_id: str,
    temporary_root: Path,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    holder = tempfile.TemporaryDirectory(
        prefix=f"bird-{db_id}-",
        dir=temporary_root,
    )
    member = _database_member(archive, db_id)
    database = Path(holder.name) / member
    database.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    cache_window = 64 * 1024 * 1024
    try:
        with archive.open(member) as source, database.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
                copied += len(chunk)
                if copied % cache_window < len(chunk):
                    target.flush()
                    _drop_file_cache(target.fileno(), copied)
                    if archive.fp is not None:
                        _drop_file_cache(
                            archive.fp.fileno(),
                            archive.fp.tell(),
                        )
            target.flush()
            os.fsync(target.fileno())
            _drop_file_cache(target.fileno(), copied)
    except BaseException:
        holder.cleanup()
        raise
    if not database.is_file():
        holder.cleanup()
        raise FileNotFoundError(database)
    return holder, database


def _outcome_summary(outcomes: dict[str, dict[str, Any]]) -> dict[str, object]:
    reasons: Counter[str] = Counter()
    eligible = 0
    for outcome in outcomes.values():
        if outcome.get("eligible") is True:
            eligible += 1
        else:
            raw = outcome.get("reasons")
            if isinstance(raw, list):
                reasons.update(str(reason) for reason in raw)
    return {
        "screened": len(outcomes),
        "eligible": eligible,
        "ineligible": len(outcomes) - eligible,
        "ineligible_reasons": dict(sorted(reasons.items())),
    }


def screen_cases(
    *,
    annotations_path: Path,
    screen_manifest_path: Path,
    database_archive_path: Path,
    temporary_root: Path,
    output_path: Path,
    resource_limit_databases: frozenset[str] = frozenset(),
) -> None:
    annotations = _load_json(annotations_path)
    manifest = _load_json(screen_manifest_path)
    raw_cases = manifest.get("cases")
    if not isinstance(annotations, list) or not isinstance(raw_cases, list):
        raise ValueError("Invalid annotations or screen manifest")
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in raw_cases:
        if not isinstance(entry, dict):
            raise ValueError("Screen manifest case is not an object")
        by_database[str(entry["database_id"])].append(entry)

    outcomes: dict[str, dict[str, Any]] = {}
    if output_path.is_file():
        existing = _load_json(output_path)
        raw_outcomes = existing.get("cases")
        if isinstance(raw_outcomes, dict):
            outcomes.update(raw_outcomes)

    unknown_resource_limits = resource_limit_databases - set(by_database)
    if unknown_resource_limits:
        raise ValueError(
            "Unknown resource-limit databases: "
            + ", ".join(sorted(unknown_resource_limits))
        )
    for db_id in sorted(resource_limit_databases):
        for entry in by_database[db_id]:
            case_id = str(entry["case_id"])
            outcomes[case_id] = {
                "eligible": False,
                "database_id": db_id,
                "annotation_index": int(entry["annotation_index"]),
                "reasons": ["database_resource_limit"],
            }

    temporary_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(database_archive_path) as archive:
        for database_index, db_id in enumerate(sorted(by_database), start=1):
            entries = by_database[db_id]
            if all(str(entry["case_id"]) in outcomes for entry in entries):
                print(
                    f"screen {database_index}/{len(by_database)} "
                    f"{db_id}: resume-skip",
                    flush=True,
                )
                continue
            holder: tempfile.TemporaryDirectory[str] | None = None
            try:
                holder, database = _extract_database(
                    archive,
                    db_id=db_id,
                    temporary_root=temporary_root,
                )
                registry = ToolRegistry(database)
                list_result = _dispatch(
                    registry,
                    call_id=f"screen_{db_id}_list",
                    name="list_tables",
                    arguments={},
                )
                tables = _listed_tables(list_result)
                table_map = {table.casefold(): table for table in tables}
                for entry in entries:
                    case_id = str(entry["case_id"])
                    if case_id in outcomes:
                        continue
                    index = int(entry["annotation_index"])
                    row = annotations[index]
                    sql = str(row["SQL"])
                    features = analyze_sql(sql)
                    parsed = {
                        str(table).casefold()
                        for table in features["physical_tables"]
                    }
                    reasons: list[str] = []
                    if not parsed:
                        reasons.append("no_physical_tables")
                    if not parsed <= set(table_map):
                        reasons.append("physical_table_not_listed")
                    runtime: set[str] = set()
                    try:
                        runtime = {
                            table.casefold()
                            for table in _runtime_physical_tables(
                                database, sql
                            )
                        }
                    except (sqlite3.Error, ValueError):
                        reasons.append("sqlite_compile_error")
                    if runtime and parsed != runtime:
                        reasons.append("parsed_runtime_table_mismatch")
                    validation_success = False
                    execution_success = False
                    truncated = False
                    returned_rows = 0
                    try:
                        _dispatch(
                            registry,
                            call_id=f"{case_id}_screen_validate",
                            name="validate_sql",
                            arguments={"sql": sql},
                        )
                        validation_success = True
                    except ValueError:
                        reasons.append("validate_sql_failed")
                    try:
                        execute_result = _dispatch(
                            registry,
                            call_id=f"{case_id}_screen_execute",
                            name="execute_sql",
                            arguments={"sql": sql},
                        )
                        execution = registry.execution_result(execute_result)
                        if execution is None:
                            reasons.append("missing_execution_result")
                        else:
                            execution_success = bool(
                                execution.execution_success
                            )
                            truncated = bool(execution.truncated)
                            returned_rows = execution.returned_row_count
                            if truncated:
                                reasons.append("execution_truncated")
                    except ValueError:
                        reasons.append("execute_sql_failed")
                    outcomes[case_id] = {
                        "eligible": not reasons,
                        "database_id": db_id,
                        "annotation_index": index,
                        "parsed_physical_tables": sorted(parsed),
                        "runtime_physical_tables": sorted(runtime),
                        "list_tables_count": len(tables),
                        "validation_success": validation_success,
                        "execution_success": execution_success,
                        "execution_truncated": truncated,
                        "returned_rows": returned_rows,
                        "reasons": sorted(set(reasons)),
                    }
            except Exception as error:
                for entry in entries:
                    case_id = str(entry["case_id"])
                    if case_id not in outcomes:
                        outcomes[case_id] = {
                            "eligible": False,
                            "database_id": db_id,
                            "annotation_index": int(
                                entry["annotation_index"]
                            ),
                            "reasons": [
                                f"database_error:{type(error).__name__}"
                            ],
                        }
            finally:
                if holder is not None:
                    holder.cleanup()

            payload = {
                "version": 1,
                "annotations": {
                    "path": str(annotations_path),
                    "sha256": sha256_file(annotations_path),
                },
                "screen_manifest": {
                    "path": str(screen_manifest_path),
                    "sha256": sha256_file(screen_manifest_path),
                },
                "database_archive": {
                    "path": str(database_archive_path),
                    "size_bytes": database_archive_path.stat().st_size,
                },
                "temporary_database_policy": (
                    "Extract one SQLite database and delete it after its group."
                ),
                "cases": outcomes,
                "summary": _outcome_summary(outcomes),
            }
            _atomic_json(output_path, payload)
            db_eligible = sum(
                outcomes[str(entry["case_id"])].get("eligible") is True
                for entry in entries
            )
            print(
                f"screen {database_index}/{len(by_database)} "
                f"{db_id}: {db_eligible}/{len(entries)} eligible",
                flush=True,
            )

    final_payload = {
        "version": 1,
        "annotations": {
            "path": str(annotations_path),
            "sha256": sha256_file(annotations_path),
        },
        "screen_manifest": {
            "path": str(screen_manifest_path),
            "sha256": sha256_file(screen_manifest_path),
        },
        "database_archive": {
            "path": str(database_archive_path),
            "size_bytes": database_archive_path.stat().st_size,
        },
        "temporary_database_policy": (
            "Extract one SQLite database and delete it after its group."
        ),
        "resource_limit_databases": sorted(resource_limit_databases),
        "cases": outcomes,
        "summary": _outcome_summary(outcomes),
    }
    _atomic_json(output_path, final_payload)

    expected = {str(entry["case_id"]) for entry in raw_cases}
    if set(outcomes) != expected:
        missing = sorted(expected - set(outcomes))
        raise ValueError(f"Eligibility screen incomplete: {missing[:5]}")
    print(json.dumps(_outcome_summary(outcomes), sort_keys=True))


def _inspect_groups(
    registry: ToolRegistry,
    *,
    case_id: str,
    physical_tables: list[str],
    char_limit: int,
) -> tuple[list[list[str]], bool, list[int]]:
    result = _dispatch(
        registry,
        call_id=f"{case_id}_inspect_probe",
        name="inspect_schema",
        arguments={"table_names": physical_tables},
    )
    full_size = len(result.model_dump_json())
    if full_size <= char_limit or len(physical_tables) <= 1:
        return [physical_tables], False, [full_size]
    groups = [[table] for table in physical_tables]
    sizes = []
    for index, group in enumerate(groups, start=1):
        part = _dispatch(
            registry,
            call_id=f"{case_id}_inspect_probe_{index}",
            name="inspect_schema",
            arguments={"table_names": group},
        )
        sizes.append(len(part.model_dump_json()))
    return groups, True, sizes


def _tool_sequence(sample: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in sample["messages"]:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls", []):
            names.append(str(call["function"]["name"]))
    return names


def _audit_messages(
    *,
    sample: dict[str, Any],
    question: str,
    evidence: str,
    gold_sql: str,
) -> list[str]:
    failures: list[str] = []
    evidence_value = evidence.strip()
    evidence_is_detectable = bool(evidence_value) and (
        evidence_value.casefold()
        not in {"true", "false", "yes", "no", "null", "none", "n/a"}
    )
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["empty_trajectory"]
    users = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(users) != 1 or users[0].get("content") != question:
        failures.append("user_message_not_exact_question")

    pending: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            failures.append("non_object_message")
            continue
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls", []):
                call_id = call.get("id")
                if not isinstance(call_id, str):
                    failures.append("tool_call_without_id")
                else:
                    pending.append(call_id)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not pending or call_id != pending.pop(0):
                failures.append("tool_result_pairing_mismatch")
            content = str(message.get("content") or "")
            if gold_sql.strip() and gold_sql in content:
                failures.append("gold_sql_in_tool_observation")
            if evidence_is_detectable and evidence_value in content:
                failures.append("evidence_in_tool_observation")
        elif role == "system":
            content = str(message.get("content") or "")
            if gold_sql.strip() and gold_sql in content:
                failures.append("gold_sql_in_system")
            if evidence_is_detectable and evidence_value in content:
                failures.append("evidence_in_system")
    if pending:
        failures.append("unpaired_tool_call")
    sequence = _tool_sequence(sample)
    if not sequence or sequence[0] != "list_tables":
        failures.append("trajectory_does_not_start_list_tables")
    if "execute_sql" not in sequence:
        failures.append("trajectory_missing_execute_sql")
    return sorted(set(failures))


def _percentiles(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)

    def at(fraction: float) -> int:
        index = min(
            len(ordered) - 1,
            max(0, math.ceil(len(ordered) * fraction) - 1),
        )
        return ordered[index]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _token_audit(
    *,
    samples: list[dict[str, Any]],
    model: Path,
    max_length: int,
    readable_count: int,
    split: str,
) -> tuple[dict[str, object], list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        trust_remote_code=False,
    )
    failures: list[str] = []
    all_turns = []
    by_case: dict[str, list[Any]] = {}
    for sample in samples:
        metadata = sample["metadata"]
        case_id = str(metadata["case_id"])
        try:
            turns = process_trajectory(
                tokenizer=tokenizer,
                sample=sample,
                split=split,
                max_length=max_length,
            )
            by_case[case_id] = turns
            all_turns.extend(turns)
            for turn in turns:
                if turn.original_sequence_tokens > max_length:
                    failures.append(f"{case_id}:original_length_over_limit")
                if turn.context_tokens_supervised != 0:
                    failures.append(f"{case_id}:context_supervised")
                if turn.supervised_tokens == 0:
                    failures.append(f"{case_id}:zero_supervised")
                if not turn.target_has_turn_end:
                    failures.append(f"{case_id}:missing_turn_end")
                if (
                    turn.target_type == "tool_call"
                    and not turn.tool_call_has_loss
                ):
                    failures.append(f"{case_id}:tool_call_without_loss")
                if (
                    turn.target_type == "final_answer"
                    and not turn.final_answer_has_loss
                ):
                    failures.append(f"{case_id}:final_answer_without_loss")
        except Exception as error:
            failures.append(
                f"{case_id}:{type(error).__name__}:{error}"
            )

    selected_samples = [
        samples[
            round(index * (len(samples) - 1) / max(1, readable_count - 1))
        ]
        for index in range(min(readable_count, len(samples)))
    ]
    readable: list[dict[str, object]] = []
    for sample in selected_samples:
        case_id = str(sample["metadata"]["case_id"])
        turns = by_case.get(case_id, [])
        readable.append(
            {
                "case_id": case_id,
                "database_id": sample["metadata"]["database_id"],
                "tool_sequence": sample["metadata"]["tool_sequence"],
                "turns": [
                    {
                        "turn_index": turn.turn_index,
                        "target_type": turn.target_type,
                        "sequence_tokens": turn.sequence_tokens,
                        "prompt_masked_tokens": turn.prompt_masked_tokens,
                        "supervised_tokens": turn.supervised_tokens,
                        "context_tokens_supervised": (
                            turn.context_tokens_supervised
                        ),
                        "tool_call_has_loss": turn.tool_call_has_loss,
                        "final_answer_has_loss": (
                            turn.final_answer_has_loss
                        ),
                        "target_prefix": turn.target_decode[:240],
                    }
                    for turn in turns
                ],
            }
        )
    if not all_turns:
        failures.append("no_assistant_turns")
        return {"readable_cases": readable}, failures
    return {
        "assistant_turns": len(all_turns),
        "sequence_tokens": _percentiles(
            [turn.sequence_tokens for turn in all_turns]
        ),
        "original_sequence_tokens": _percentiles(
            [turn.original_sequence_tokens for turn in all_turns]
        ),
        "supervised_tokens": _percentiles(
            [turn.supervised_tokens for turn in all_turns]
        ),
        "prompt_masked_tokens": _percentiles(
            [turn.prompt_masked_tokens for turn in all_turns]
        ),
        "tool_call_targets": sum(
            turn.target_type == "tool_call" for turn in all_turns
        ),
        "final_answer_targets": sum(
            turn.target_type == "final_answer" for turn in all_turns
        ),
        "readable_cases": readable,
    }, failures


def build_sft(
    *,
    annotations_path: Path,
    pool_manifest_path: Path,
    database_archive_path: Path,
    temporary_root: Path,
    output_path: Path,
    tools_output_path: Path,
    audit_output_path: Path,
    model: Path,
    max_length: int,
    schema_observation_char_limit: int,
    readable_audit_count: int,
    split: str,
    expected_cases: int,
    dataset_name: str,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    annotations = _load_json(annotations_path)
    manifest = _load_json(pool_manifest_path)
    raw_cases = manifest.get("cases")
    if not isinstance(annotations, list) or not isinstance(raw_cases, list):
        raise ValueError("Invalid annotations or SFT pool manifest")
    if len(raw_cases) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} {split} cases, got {len(raw_cases)}"
        )

    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in raw_cases:
        by_database[str(entry["database_id"])].append(entry)
    temporary_root.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []
    final_tools: list[dict[str, object]] | None = None
    schema_split_cases = 0
    schema_observation_sizes: list[int] = []

    with zipfile.ZipFile(database_archive_path) as archive:
        for database_index, db_id in enumerate(sorted(by_database), start=1):
            entries = by_database[db_id]
            holder: tempfile.TemporaryDirectory[str] | None = None
            try:
                holder, database = _extract_database(
                    archive,
                    db_id=db_id,
                    temporary_root=temporary_root,
                )
                registry = ToolRegistry(database)
                list_result = _dispatch(
                    registry,
                    call_id=f"build_{db_id}_list",
                    name="list_tables",
                    arguments={},
                )
                tables = _listed_tables(list_result)
                table_map = {table.casefold(): table for table in tables}
                case_specs: list[dict[str, Any]] = []
                metadata_by_case: dict[str, dict[str, Any]] = {}
                db_failed = False
                for entry in entries:
                    case_id = str(entry["case_id"])
                    index = int(entry["annotation_index"])
                    row = annotations[index]
                    question = str(row["question"])
                    sql = str(row["SQL"])
                    evidence = str(row.get("evidence", ""))
                    features = analyze_sql(sql)
                    parsed = {
                        str(table).casefold()
                        for table in features["physical_tables"]
                    }
                    case_failures: list[str] = []
                    if not parsed or not parsed <= set(table_map):
                        case_failures.append(
                            "gold_physical_tables_not_subset_list_tables"
                        )
                    actual_tables = [
                        table_map[name] for name in sorted(parsed)
                        if name in table_map
                    ]
                    try:
                        runtime = {
                            table.casefold()
                            for table in _runtime_physical_tables(
                                database, sql
                            )
                        }
                        if runtime != parsed:
                            case_failures.append(
                                "parsed_runtime_physical_tables_differ"
                            )
                    except (sqlite3.Error, ValueError):
                        case_failures.append("runtime_table_resolution_failed")

                    groups: list[list[str]] = []
                    split_schema = False
                    sizes: list[int] = []
                    if not case_failures:
                        try:
                            groups, split_schema, sizes = _inspect_groups(
                                registry,
                                case_id=case_id,
                                physical_tables=actual_tables,
                                char_limit=schema_observation_char_limit,
                            )
                            schema_observation_sizes.extend(sizes)
                            if split_schema:
                                schema_split_cases += 1
                        except ValueError:
                            case_failures.append(
                                "schema_observation_failed"
                            )

                    turns: list[dict[str, object]] = [
                        {"name": "list_tables", "arguments": {}}
                    ]
                    for group in groups:
                        turns.append(
                            {
                                "name": "inspect_schema",
                                "arguments": {"table_names": group},
                            }
                        )
                    if bool(features["complex"]):
                        turns.append(
                            {"name": "validate_sql", "arguments": {}}
                        )
                    turns.append({"name": "execute_sql", "arguments": {}})

                    if not case_failures:
                        try:
                            for turn_index, turn in enumerate(
                                turns, start=1
                            ):
                                name = str(turn["name"])
                                arguments = dict(turn["arguments"])
                                if name in {"validate_sql", "execute_sql"}:
                                    arguments["sql"] = sql
                                result = _dispatch(
                                    registry,
                                    call_id=(
                                        f"{case_id}_preflight_"
                                        f"{turn_index}_{name}"
                                    ),
                                    name=name,
                                    arguments=arguments,
                                )
                                if name == "execute_sql":
                                    execution = registry.execution_result(
                                        result
                                    )
                                    if (
                                        execution is None
                                        or not execution.execution_success
                                        or execution.truncated
                                    ):
                                        case_failures.append(
                                            "invalid_execution_result"
                                        )
                        except ValueError:
                            case_failures.append(
                                "real_tool_preflight_failed"
                            )

                    if case_failures:
                        db_failed = True
                        failures.append(
                            {
                                "case_id": case_id,
                                "database_id": db_id,
                                "reasons": sorted(set(case_failures)),
                            }
                        )
                        continue
                    case_specs.append(
                        {
                            "case_id": case_id,
                            "question": question,
                            "sql": sql,
                            "inspect_tables": actual_tables,
                            "tool_turns": turns,
                            "answer": {"kind": "execution_json"},
                            "template_family": (
                                "bird_complex"
                                if features["complex"]
                                else "bird_simple_single_table"
                            ),
                            "difficulty": "unavailable",
                            "parameter_signature": (
                                f"{db_id}:{index}"
                            ),
                        }
                    )
                    metadata_by_case[case_id] = {
                        "database_id": db_id,
                        "annotation_index": index,
                        "evidence": evidence,
                        "structural_features": features,
                        "gold_physical_tables": actual_tables,
                        "schema_observation_split": split_schema,
                        "schema_observation_sizes": sizes,
                        "observations_source": "Harness v2 real tools",
                    }

                if db_failed:
                    continue
                with tempfile.TemporaryDirectory(
                    prefix=f"bird-build-{db_id}-",
                    dir=temporary_root,
                ) as build_dir_raw:
                    build_dir = Path(build_dir_raw)
                    cases_path = build_dir / "cases.json"
                    db_output = build_dir / "samples.jsonl"
                    db_tools = build_dir / "tools.json"
                    _atomic_json(
                        cases_path,
                        {
                            "dataset_name": dataset_name,
                            "cases": case_specs,
                        },
                    )
                    samples = build_dataset(
                        database=database,
                        cases_path=cases_path,
                        output=db_output,
                        tools_output=db_tools,
                        split=split,
                    )
                    tools = _load_json(db_tools)
                    if final_tools is None:
                        final_tools = tools
                    elif tools != final_tools:
                        raise ValueError(
                            f"Tool schemas differ for database {db_id}"
                        )
                    for sample in samples:
                        metadata = sample["metadata"]
                        case_id = str(metadata["case_id"])
                        metadata.update(metadata_by_case[case_id])
                        row = annotations[
                            int(metadata["annotation_index"])
                        ]
                        message_failures = _audit_messages(
                            sample=sample,
                            question=str(row["question"]),
                            evidence=str(row.get("evidence", "")),
                            gold_sql=str(row["SQL"]),
                        )
                        if message_failures:
                            failures.append(
                                {
                                    "case_id": case_id,
                                    "database_id": db_id,
                                    "reasons": message_failures,
                                }
                            )
                        generated.append(sample)
            except Exception as error:
                failures.append(
                    {
                        "case_id": None,
                        "database_id": db_id,
                        "reasons": [
                            f"database_build_error:{type(error).__name__}:{error}"
                        ],
                    }
                )
            finally:
                if holder is not None:
                    holder.cleanup()
            print(
                f"build {database_index}/{len(by_database)} {db_id}: "
                f"{len(entries)} cases",
                flush=True,
            )

    audit: dict[str, object] = {
        "version": 1,
        "dataset_name": dataset_name,
        "split": split,
        "status": "failed" if failures else "preflight_complete",
        "inputs": {
            "annotations": {
                "path": str(annotations_path),
                "sha256": sha256_file(annotations_path),
            },
            "pool_manifest": {
                "path": str(pool_manifest_path),
                "sha256": sha256_file(pool_manifest_path),
            },
            "database_archive": {
                "path": str(database_archive_path),
                "size_bytes": database_archive_path.stat().st_size,
            },
        },
        "expected_cases": len(raw_cases),
        "generated_before_token_audit": len(generated),
        "preflight_failures": len(failures),
        "failure_examples": failures[:50],
        "schema_observation_char_limit": schema_observation_char_limit,
        "schema_split_cases": schema_split_cases,
        "schema_observation_chars": (
            _percentiles(schema_observation_sizes)
            if schema_observation_sizes
            else None
        ),
        "temporary_database_policy": (
            "Extract one SQLite database, generate its cases, then delete it."
        ),
    }
    if failures or len(generated) != len(raw_cases):
        _atomic_json(audit_output_path, audit)
        raise ValueError(
            "Hard preflight failed; final training JSONL was not generated"
        )

    token_report, token_failures = _token_audit(
        samples=generated,
        model=model,
        max_length=max_length,
        readable_count=readable_audit_count,
        split=split,
    )
    audit["assistant_only_preprocessing"] = token_report
    audit["token_failures"] = len(token_failures)
    audit["token_failure_examples"] = token_failures[:50]
    audit["trajectory_paths"] = dict(
        sorted(
            Counter(
                " -> ".join(_tool_sequence(sample))
                for sample in generated
            ).items()
        )
    )
    audit["database_counts"] = dict(
        sorted(
            Counter(
                str(sample["metadata"]["database_id"])
                for sample in generated
            ).items()
        )
    )
    if token_failures:
        audit["status"] = "failed"
        _atomic_json(audit_output_path, audit)
        raise ValueError(
            "Assistant-only token audit failed; final JSONL not generated"
        )

    staging = output_path.with_suffix(output_path.suffix + ".staging")
    if staging.exists():
        staging.unlink()
    _atomic_jsonl(staging, generated)
    if final_tools is None:
        raise ValueError("No tool schemas were generated")
    _atomic_json(tools_output_path, final_tools)
    staging.replace(output_path)
    audit["status"] = "ready"
    audit["output"] = {
        "path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "cases": len(generated),
    }
    audit["tools"] = {
        "path": str(tools_output_path),
        "sha256": sha256_file(tools_output_path),
    }
    _atomic_json(audit_output_path, audit)
    print(json.dumps(audit["output"], sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen or build multi-database BIRD SFT trajectories."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    screen = subparsers.add_parser("screen")
    screen.add_argument("--annotations", type=Path, required=True)
    screen.add_argument("--screen-manifest", type=Path, required=True)
    screen.add_argument("--database-archive", type=Path, required=True)
    screen.add_argument("--temporary-root", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument(
        "--resource-limit-database",
        action="append",
        default=[],
        help=(
            "Mark every case in this database ineligible when the current "
            "CPU instance cannot safely execute it; may be repeated."
        ),
    )

    build = subparsers.add_parser("build")
    build.add_argument("--annotations", type=Path, required=True)
    build.add_argument("--pool-manifest", type=Path, required=True)
    build.add_argument("--database-archive", type=Path, required=True)
    build.add_argument("--temporary-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--tools-output", type=Path, required=True)
    build.add_argument("--audit-output", type=Path, required=True)
    build.add_argument("--model", type=Path, required=True)
    build.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    build.add_argument(
        "--schema-observation-char-limit",
        type=int,
        default=DEFAULT_SCHEMA_OBSERVATION_CHAR_LIMIT,
    )
    build.add_argument("--readable-audit-count", type=int, default=8)
    build.add_argument("--split", choices=("train", "dev"), default="train")
    build.add_argument("--expected-cases", type=int, default=2500)
    build.add_argument(
        "--dataset-name",
        default="bird_full_train_sft_v1",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "screen":
        screen_cases(
            annotations_path=args.annotations,
            screen_manifest_path=args.screen_manifest,
            database_archive_path=args.database_archive,
            temporary_root=args.temporary_root,
            output_path=args.output,
            resource_limit_databases=frozenset(
                args.resource_limit_database
            ),
        )
    else:
        build_sft(
            annotations_path=args.annotations,
            pool_manifest_path=args.pool_manifest,
            database_archive_path=args.database_archive,
            temporary_root=args.temporary_root,
            output_path=args.output,
            tools_output_path=args.tools_output,
            audit_output_path=args.audit_output,
            model=args.model,
            max_length=args.max_length,
            schema_observation_char_limit=(
                args.schema_observation_char_limit
            ),
            readable_audit_count=args.readable_audit_count,
            split=args.split,
            expected_cases=args.expected_cases,
            dataset_name=args.dataset_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
