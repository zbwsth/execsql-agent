"""Export verl-independent GRPO samples to the verl v0.9.0 Parquet boundary."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb

from execsql_agent.rlvr.models import GRPOSample

VERL_TARGET_VERSION = "v0.9.0"
VERL_RECORD_KEYS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}

_CREATE_TABLE_SQL = """
CREATE TABLE verl_records (
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
            database_sha256 VARCHAR
        ),
        dataset STRUCT(
            dataset_name VARCHAR,
            difficulty VARCHAR,
            evaluation_group VARCHAR,
            tags VARCHAR[],
            prompt_version VARCHAR,
            comparator_version VARCHAR
        )
    ) NOT NULL
)
"""


def _contains_key(value: object, forbidden: str) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() == forbidden.casefold() or _contains_key(child, forbidden)
            for key, child in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def sample_to_verl_record(
    sample: GRPOSample,
    *,
    index: int,
    split: str,
) -> dict[str, object]:
    """Map one validated sample to the fields consumed by verl v0.9.0 RLHFDataset."""

    if index < 0:
        raise ValueError("index must be non-negative")
    if not split:
        raise ValueError("split must not be empty")

    prompt = [message.model_dump(mode="json") for message in sample.messages]
    prompt_text = json.dumps(prompt, ensure_ascii=False)
    if "expected_result" in prompt_text.casefold() or "gold_sql" in prompt_text.casefold():
        raise ValueError("private verifier data must not enter the verl prompt")

    private_ground_truth = {
        "verifier_type": sample.private_verifier_metadata.verifier_type,
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
        },
    }
    if set(record) != VERL_RECORD_KEYS:
        raise AssertionError("unexpected verl record shape")
    if _contains_key(record, "gold_sql"):
        raise ValueError("gold_sql must never be exported")
    return record


def _duckdb_string_literal(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def write_verl_parquet(
    samples: Sequence[GRPOSample],
    output_path: str | Path,
    *,
    split: str,
) -> Path:
    """Atomically write nested v0.9.0-compatible records without importing verl."""

    if not samples:
        raise ValueError("at least one GRPO sample is required")
    output = Path(output_path).resolve()
    if output.suffix.casefold() != ".parquet":
        raise ValueError("verl dataset output must use a .parquet suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)

    connection = duckdb.connect(":memory:")
    try:
        connection.execute(_CREATE_TABLE_SQL)
        for index, sample in enumerate(samples):
            record = sample_to_verl_record(sample, index=index, split=split)
            connection.execute(
                "INSERT INTO verl_records VALUES (?, ?, ?, ?, ?)",
                [
                    record["data_source"],
                    record["prompt"],
                    record["ability"],
                    record["reward_model"],
                    record["extra_info"],
                ],
            )
        connection.execute(
            "COPY verl_records TO "
            f"{_duckdb_string_literal(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()

    os.replace(temporary, output)
    return output


def read_intermediate_jsonl(path: str | Path) -> list[GRPOSample]:
    """Load validated Phase 1 JSONL without accepting arbitrary record fields."""

    source = Path(path)
    samples: list[GRPOSample] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                samples.append(GRPOSample.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"Invalid GRPO sample at line {line_number}: {error}") from error
    if not samples:
        raise ValueError(f"No GRPO samples found in {source}")
    return samples


def read_verl_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Read exported rows for CPU-side inspection and tests."""

    connection = duckdb.connect(":memory:")
    try:
        cursor = connection.execute(
            "SELECT data_source, prompt, ability, reward_model, extra_info "
            'FROM read_parquet(?) ORDER BY extra_info."index"',
            [str(Path(path).resolve())],
        )
        columns = [str(item[0]) for item in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Export Phase 1 GRPO JSONL for verl {VERL_TARGET_VERSION}."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = read_intermediate_jsonl(args.input)
    output = write_verl_parquet(samples, args.output, split=args.split)
    print(f"Exported {len(samples)} {args.split} samples for {VERL_TARGET_VERSION}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
