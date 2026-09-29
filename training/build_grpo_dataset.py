"""Build a verl-independent GRPO dataset from explicit evaluation inputs."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from execsql_agent.config import load_domain_config
from execsql_agent.evaluation.synthetic import load_evaluation_dataset
from execsql_agent.models import EvaluationDataset
from execsql_agent.rlvr.models import (
    DatabaseMetadata,
    ExtraInfo,
    GRPOSample,
    PromptMessage,
    VerifierMetadata,
)
from execsql_agent.tools.schema_loader import SchemaLoader

SYSTEM_INSTRUCTION = """You generate exactly one safe, read-only SQLite query.
Use only tables and columns in the supplied schema and follow the domain context.
Return JSON only in exactly this shape: {"sql":"<single read-only SQL statement>"}
Do not return Markdown, prose, tool calls, database paths, or multiple SQL statements."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_messages(
    *, question: str, schema_context: str, domain_context: str
) -> list[PromptMessage]:
    """Build model input from an explicit allowlist of public fields."""

    system = (
        f"{SYSTEM_INSTRUCTION}\n\nSQLite schema:\n{schema_context.strip()}"
        f"\n\nDomain context:\n{domain_context.strip()}"
    )
    return [
        PromptMessage(role="system", content=system),
        PromptMessage(role="user", content=question),
    ]


def build_samples(
    dataset: EvaluationDataset,
    *,
    database_path: Path,
    database_sha256: str,
    schema_context: str,
    domain_context: str,
) -> list[GRPOSample]:
    """Project evaluation cases without exposing gold SQL to the model."""

    samples: list[GRPOSample] = []
    for case in dataset.cases:
        samples.append(
            GRPOSample(
                sample_id=case.id,
                data_source=dataset.dataset_name,
                question=case.question,
                messages=build_messages(
                    question=case.question,
                    schema_context=schema_context,
                    domain_context=domain_context,
                ),
                database_metadata=DatabaseMetadata(
                    database_id=case.database_id,
                    database_path=str(database_path.resolve()),
                    database_sha256=database_sha256,
                ),
                private_verifier_metadata=VerifierMetadata(
                    expected_result=case.expected_result
                ),
                extra_info=ExtraInfo(
                    dataset_name=dataset.dataset_name,
                    difficulty=case.difficulty,
                    evaluation_group=case.evaluation_group,
                    tags=list(case.tags),
                ),
            )
        )
    return samples


def load_samples(
    *,
    dataset_path: Path,
    database_path: Path,
    domain_config_path: Path | None = None,
) -> list[GRPOSample]:
    """Load explicit inputs after verifying the trusted SQLite database."""

    database_path = database_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database_path}")
    dataset = load_evaluation_dataset(dataset_path)
    schema = SchemaLoader(database_path).load()
    unexpected_ids = sorted(
        {
            case.database_id
            for case in dataset.cases
            if case.database_id != schema.database_id
        }
    )
    if unexpected_ids:
        raise ValueError(
            f"Dataset database_id values do not match "
            f"{schema.database_id}: {unexpected_ids}"
        )
    domain_context = "No additional domain context."
    if domain_config_path is not None:
        domain_context = load_domain_config(domain_config_path).to_prompt()
    return build_samples(
        dataset,
        database_path=database_path,
        database_sha256=_sha256(database_path),
        schema_context=schema.summary_text,
        domain_context=domain_context,
    )


def write_jsonl(
    samples: Sequence[GRPOSample], output_path: Path
) -> Path:
    """Write one validated intermediate sample per UTF-8 JSONL line."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            stream.write(sample.model_dump_json())
            stream.write("\n")
    return output_path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build leak-resistant SQL GRPO intermediate data from "
            "explicit dataset and database paths."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--domain-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = load_samples(
        dataset_path=args.dataset,
        database_path=args.database,
        domain_config_path=args.domain_config,
    )
    output = write_jsonl(samples, args.output)
    print(f"Generated {len(samples)} GRPO samples: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
