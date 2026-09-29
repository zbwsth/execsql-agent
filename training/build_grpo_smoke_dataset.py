"""Build deterministic DEV/SMOKE ONLY SQLite and verl Parquet assets."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from execsql_agent.models import ExpectedResult
from execsql_agent.rlvr.models import (
    DatabaseMetadata,
    ExtraInfo,
    GRPOSample,
    VerifierMetadata,
)
from execsql_agent.tools.schema_loader import SchemaLoader
from execsql_agent.tools.sql_executor import SQLExecutor
from scripts.create_demo_database import create_demo_database
from training.build_grpo_dataset import build_messages
from training.verl_grpo_adapter import write_verl_parquet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/grpo_smoke"
SMOKE_DATA_SOURCE = "execsql_sql_dev_smoke_v1"

SMOKE_DOMAIN_CONTEXT = """DEV / SMOKE ONLY.
This deterministic e-commerce database exists only to exercise the GRPO plumbing.
It is not a formal benchmark and must never be reported as a formal result.
Dates are ISO-8601 text. Monetary values are stored as SQLite REAL values."""

SMOKE_OUTPUT_CONTRACT = """Output contract: Your entire response must be exactly
one JSON object on one line. The first character must be { and the last character
must be }. Never output Markdown fences, the word json outside the object,
analysis, explanation, or any other text."""

DIAGNOSTIC_SAMPLE_ID = "smoke_sql_008"


@dataclass(frozen=True)
class SmokeCase:
    sample_id: str
    question: str
    reference_sql: str
    ordered: bool = True


@dataclass(frozen=True)
class SmokeBuildResult:
    database_path: Path
    train_path: Path
    val_path: Path
    diagnostic_path: Path
    train_count: int
    val_count: int


SMOKE_CASES = (
    SmokeCase(
        "smoke_sql_001",
        "How many customers are in the database?",
        "SELECT COUNT(*) AS customer_count FROM customers",
    ),
    SmokeCase(
        "smoke_sql_002",
        "How many products are in the computer accessories category?",
        "SELECT COUNT(*) AS product_count FROM products WHERE category = '电脑配件'",
    ),
    SmokeCase(
        "smoke_sql_003",
        "List the three most expensive products from most to least expensive.",
        "SELECT product_name, unit_price FROM products "
        "ORDER BY unit_price DESC, product_id ASC LIMIT 3",
    ),
    SmokeCase(
        "smoke_sql_004",
        "For each product category, how many products does it contain?",
        "SELECT category, COUNT(*) AS product_count FROM products "
        "GROUP BY category ORDER BY category ASC",
    ),
    SmokeCase(
        "smoke_sql_005",
        "How many orders have completed status?",
        "SELECT COUNT(*) AS order_count FROM orders WHERE status = 'completed'",
    ),
    SmokeCase(
        "smoke_sql_006",
        "What is the total amount of all completed orders?",
        "SELECT ROUND(SUM(total_amount), 2) AS completed_total FROM orders "
        "WHERE status = 'completed'",
    ),
    SmokeCase(
        "smoke_sql_007",
        "Show every order status and its order count, sorted by status.",
        "SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY status ASC",
    ),
    SmokeCase(
        "smoke_sql_008",
        "Which five customers have the highest total order amount?",
        "SELECT c.customer_name, ROUND(SUM(o.total_amount), 2) AS total_amount "
        "FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
        "GROUP BY c.customer_id, c.customer_name "
        "ORDER BY total_amount DESC, c.customer_id ASC LIMIT 5",
    ),
    SmokeCase(
        "smoke_sql_009",
        "What is the average product price in each category?",
        "SELECT category, ROUND(AVG(unit_price), 2) AS average_price FROM products "
        "GROUP BY category ORDER BY category ASC",
    ),
    SmokeCase(
        "smoke_sql_010",
        "List the first five customers to sign up, earliest first.",
        "SELECT customer_name, signup_date FROM customers "
        "ORDER BY signup_date ASC, customer_id ASC LIMIT 5",
    ),
    SmokeCase(
        "smoke_sql_011",
        "Which five products sold the most units?",
        "SELECT p.product_name, SUM(oi.quantity) AS units_sold "
        "FROM products p JOIN order_items oi ON oi.product_id = p.product_id "
        "GROUP BY p.product_id, p.product_name "
        "ORDER BY units_sold DESC, p.product_id ASC LIMIT 5",
    ),
    SmokeCase(
        "smoke_sql_012",
        "How many orders were placed on or after March 1, 2025?",
        "SELECT COUNT(*) AS order_count FROM orders WHERE order_date >= '2025-03-01'",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_database_path(output_dir: Path, database_path: Path) -> str:
    if output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        return "data/grpo_smoke/sql_smoke.db"
    return str(database_path.resolve())


def build_smoke_samples(
    database_path: Path,
    *,
    metadata_database_path: str,
) -> list[GRPOSample]:
    """Execute private reference SQL and build leak-resistant smoke samples."""

    schema = SchemaLoader(database_path).load()
    executor = SQLExecutor(database_path, max_rows=100)
    database_sha256 = _sha256(database_path)
    samples: list[GRPOSample] = []
    for case in SMOKE_CASES:
        execution = executor.execute(case.reference_sql)
        if not execution.execution_success or execution.truncated:
            raise RuntimeError(f"Smoke reference SQL failed for {case.sample_id}")
        expected = ExpectedResult(
            columns=execution.columns,
            rows=execution.rows,
            ordered=case.ordered,
        )
        samples.append(
            GRPOSample(
                sample_id=case.sample_id,
                data_source=SMOKE_DATA_SOURCE,
                question=case.question,
                messages=build_messages(
                    question=case.question,
                    schema_context=schema.summary_text,
                    domain_context=(f"{SMOKE_DOMAIN_CONTEXT}\n\n{SMOKE_OUTPUT_CONTRACT}"),
                ),
                database_metadata=DatabaseMetadata(
                    database_id=schema.database_id,
                    database_path=metadata_database_path,
                    database_sha256=database_sha256,
                ),
                private_verifier_metadata=VerifierMetadata(expected_result=expected),
                extra_info=ExtraInfo(
                    dataset_name=SMOKE_DATA_SOURCE,
                    difficulty="smoke",
                    evaluation_group="dev_smoke_only",
                    tags=["dev", "smoke_only"],
                ),
            )
        )
    return samples


def build_smoke_assets(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> SmokeBuildResult:
    """Create the database plus train, validation, and diagnostic Parquet data."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    database_path = destination / "sql_smoke.db"
    create_demo_database(database_path)
    samples = build_smoke_samples(
        database_path.resolve(),
        metadata_database_path=_metadata_database_path(destination, database_path),
    )
    train_samples = samples[:10]
    val_samples = samples[10:]
    train_path = write_verl_parquet(
        train_samples,
        destination / "train.parquet",
        split="train",
    )
    val_path = write_verl_parquet(
        val_samples,
        destination / "val.parquet",
        split="val",
    )
    diagnostic_samples = [sample for sample in samples if sample.sample_id == DIAGNOSTIC_SAMPLE_ID]
    if len(diagnostic_samples) != 1:
        raise RuntimeError(f"Expected exactly one diagnostic sample: {DIAGNOSTIC_SAMPLE_ID}")
    diagnostic_path = write_verl_parquet(
        diagnostic_samples,
        destination / "diagnostic.parquet",
        split="diagnostic",
    )
    return SmokeBuildResult(
        database_path=database_path.resolve(),
        train_path=train_path,
        val_path=val_path,
        diagnostic_path=diagnostic_path,
        train_count=len(train_samples),
        val_count=len(val_samples),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic DEV/SMOKE ONLY SQL GRPO assets."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_smoke_assets(args.output_dir)
    print("Built DEV/SMOKE ONLY SQL GRPO assets:")
    print(f"  database: {result.database_path}")
    print(f"  train: {result.train_path} ({result.train_count} rows)")
    print(f"  val: {result.val_path} ({result.val_count} rows)")
    print(f"  diagnostic: {result.diagnostic_path} (1 row)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
