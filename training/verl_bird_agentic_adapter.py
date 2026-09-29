"""VERL v0.9.0 agentic Parquet boundary for frozen BIRD GRPO cases."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import duckdb

from execsql_agent.rlvr.models import GRPOSample

VERL_TARGET_VERSION = "v0.9.0"
HARNESS_V2_MAX_STEPS = 6
HARNESS_V2_TOOL_NAMES = (
    "list_tables",
    "inspect_schema",
    "validate_sql",
    "execute_sql",
)
BIRD_GRPO_DATA_SOURCE = "bird_train_grpo_agentic_v1"
VERL_AGENTIC_RECORD_KEYS = {
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
}

HARNESS_V2_SYSTEM_PROMPT = (
    "You are a read-only SQLite analysis agent. Use only the provided tools. "
    "Inspect schema when needed, never invent tool names or fields, and give a "
    "concise final answer only after enough observations. Exploratory SQL is "
    "allowed. Before producing a database-backed final answer, the last "
    "execute_sql call must directly return exactly the complete columns and "
    "complete rows required by the user's question. Do not combine earlier SQL "
    "observations, do not omit required fields, and do not return extra columns "
    "or extra rows. Base the final answer only on the last successful execute_sql "
    "result. After a successful execute_sql result already completely answers "
    "the user question, stop calling tools and provide the final answer; never "
    "repeat the same execute_sql call. For Top-K requests, use the exact "
    "requested LIMIT in that final SQL; "
    "never fetch more rows and truncate them only in natural language."
)

_CREATE_TABLE_SQL = """
CREATE TABLE verl_agentic_records (
    data_source VARCHAR NOT NULL,
    prompt STRUCT(role VARCHAR, content VARCHAR)[] NOT NULL,
    ability VARCHAR NOT NULL,
    reward_model STRUCT(style VARCHAR, ground_truth VARCHAR) NOT NULL,
    extra_info STRUCT(
        "index" BIGINT,
        split VARCHAR,
        sample_id VARCHAR,
        database STRUCT(
            database_id VARCHAR,
            database_path VARCHAR,
            database_sha256 VARCHAR,
            database_archive_path VARCHAR,
            database_archive_member VARCHAR
        ),
        dataset STRUCT(
            dataset_name VARCHAR,
            difficulty VARCHAR,
            evaluation_group VARCHAR,
            tags VARCHAR[],
            prompt_version VARCHAR,
            comparator_version VARCHAR
        ),
        agent_name VARCHAR,
        tool_selection VARCHAR[],
        need_tools_kwargs BOOLEAN,
        tools_kwargs STRUCT(
            list_tables STRUCT(create_kwargs STRUCT(
                database_id VARCHAR,
                database_path VARCHAR,
                database_sha256 VARCHAR,
                database_archive_member VARCHAR
            )),
            inspect_schema STRUCT(create_kwargs STRUCT(
                database_id VARCHAR,
                database_path VARCHAR,
                database_sha256 VARCHAR,
                database_archive_member VARCHAR
            )),
            validate_sql STRUCT(create_kwargs STRUCT(
                database_id VARCHAR,
                database_path VARCHAR,
                database_sha256 VARCHAR,
                database_archive_member VARCHAR
            )),
            execute_sql STRUCT(create_kwargs STRUCT(
                database_id VARCHAR,
                database_path VARCHAR,
                database_sha256 VARCHAR,
                database_archive_member VARCHAR
            ))
        )
    ) NOT NULL
)
"""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_policy_messages(question: str) -> list[dict[str, str]]:
    """Return exactly the frozen Harness-v2 system and user messages."""

    if not question.strip():
        raise ValueError("question must not be blank")
    return [
        {"role": "system", "content": HARNESS_V2_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _contains_key(value: object, forbidden: str) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() == forbidden.casefold() or _contains_key(child, forbidden)
            for key, child in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _assert_policy_boundary(
    record: Mapping[str, object],
    *,
    gold_sql: str | None = None,
    evidence: str | None = None,
) -> None:
    prompt = record.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError("record prompt must be a list")
    prompt_text = json.dumps(prompt, ensure_ascii=False).casefold()
    forbidden_markers = (
        "gold_sql",
        "expected_result",
        "database_path",
        "database_sha256",
        "database_archive",
    )
    if any(marker in prompt_text for marker in forbidden_markers):
        raise ValueError("private verifier/database metadata entered policy prompt")
    if gold_sql and gold_sql.strip().casefold() in prompt_text:
        raise ValueError("gold SQL entered policy prompt")
    if evidence and evidence.strip().casefold() in prompt_text:
        raise ValueError("BIRD evidence entered policy prompt")
    if _contains_key(record, "gold_sql") or _contains_key(record, "evidence"):
        raise ValueError("gold SQL or evidence key entered formal VERL record")


def _tool_kwargs(sample: GRPOSample) -> dict[str, dict[str, dict[str, str]]]:
    database = sample.database_metadata
    if not database.database_archive_path or not database.database_archive_member:
        raise ValueError("agentic BIRD samples require archive-backed database metadata")
    create_kwargs = {
        "database_id": database.database_id,
        "database_path": database.database_path,
        "database_sha256": database.database_sha256,
        "database_archive_member": database.database_archive_member,
    }
    return {name: {"create_kwargs": dict(create_kwargs)} for name in HARNESS_V2_TOOL_NAMES}


def sample_to_agentic_record(
    sample: GRPOSample,
    *,
    index: int,
    split: str,
    gold_sql_for_scan: str | None = None,
    evidence_for_scan: str | None = None,
) -> dict[str, object]:
    """Map one private/public sample to VERL's native ToolAgentLoop boundary."""

    if index < 0:
        raise ValueError("index must be non-negative")
    if not split:
        raise ValueError("split must not be blank")
    if sample.private_verifier_metadata.response_contract != "agentic_tool_loop_v1":
        raise ValueError("agentic record requires agentic_tool_loop_v1 verifier metadata")
    prompt = build_policy_messages(sample.question)
    private_ground_truth = {
        "verifier_type": sample.private_verifier_metadata.verifier_type,
        "response_contract": sample.private_verifier_metadata.response_contract,
        "expected_result": sample.private_verifier_metadata.expected_result.model_dump(mode="json"),
    }
    record: dict[str, object] = {
        "data_source": sample.data_source,
        "prompt": prompt,
        "ability": "sql",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(
                private_ground_truth,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
        "extra_info": {
            "index": index,
            "split": split,
            "sample_id": sample.sample_id,
            "database": sample.database_metadata.model_dump(mode="json"),
            "dataset": sample.extra_info.model_dump(mode="json"),
            "agent_name": "tool_agent",
            "tool_selection": list(HARNESS_V2_TOOL_NAMES),
            "need_tools_kwargs": True,
            "tools_kwargs": _tool_kwargs(sample),
        },
    }
    if set(record) != VERL_AGENTIC_RECORD_KEYS:
        raise AssertionError("unexpected VERL agentic record shape")
    _assert_policy_boundary(
        record,
        gold_sql=gold_sql_for_scan,
        evidence=evidence_for_scan,
    )
    return record


def _duckdb_string_literal(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def write_agentic_parquet(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    """Atomically write validated nested ToolAgentLoop records."""

    if not records:
        raise ValueError("at least one agentic record is required")
    output = Path(output_path).resolve()
    if output.suffix.casefold() != ".parquet":
        raise ValueError("VERL dataset output must use a .parquet suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)

    connection = duckdb.connect(":memory:")
    try:
        connection.execute(_CREATE_TABLE_SQL)
        for record in records:
            _assert_policy_boundary(record)
            connection.execute(
                "INSERT INTO verl_agentic_records VALUES (?, ?, ?, ?, ?)",
                [
                    record["data_source"],
                    record["prompt"],
                    record["ability"],
                    record["reward_model"],
                    record["extra_info"],
                ],
            )
        connection.execute(
            "COPY verl_agentic_records TO "
            f"{_duckdb_string_literal(temporary)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()

    os.replace(temporary, output)
    return output
