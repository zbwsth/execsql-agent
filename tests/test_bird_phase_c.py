"""Phase C BIRD data preparation tests."""

from __future__ import annotations

import json
from pathlib import Path

from training.bird_phase_c import analyze_sql, stratified_sample
from training.build_bird_sft_dataset import (
    _audit_messages,
)
from training.build_bird_sft_dataset import (
    build_parser as build_orchestration_parser,
)
from training.build_sft_dataset import build_dataset


def test_analyze_sql_excludes_cte_alias_from_physical_tables() -> None:
    features = analyze_sql(
        """
        WITH recent AS (
            SELECT customer_id, SUM(total) AS amount
            FROM orders
            GROUP BY customer_id
        )
        SELECT c.name, r.amount
        FROM recent AS r
        JOIN customers AS c ON c.id = r.customer_id
        """
    )

    assert features["physical_tables"] == ["customers", "orders"]
    assert features["physical_table_count"] == 2
    assert features["join_count"] == 1
    assert features["aggregation"] is True
    assert features["group_by"] is True
    assert features["nested_select"] is True
    assert features["cte"] is True
    assert features["complex"] is True


def test_analyze_sql_detects_set_operation() -> None:
    features = analyze_sql(
        "SELECT id FROM customers UNION SELECT customer_id FROM orders"
    )

    assert features["physical_tables"] == ["customers", "orders"]
    assert features["set_operation"] is True
    assert features["table_mode"] == "multi"


def test_stratified_sample_is_deterministic_disjoint_ready() -> None:
    records = []
    for index in range(60):
        records.append(
            {
                "case_id": f"case_{index}",
                "annotation_index": index,
                "database_id": f"db_{index % 6}",
                "features": {
                    "table_mode": "single" if index % 2 else "multi",
                    "join_bucket": str(index % 3),
                    "aggregation": bool(index % 2),
                    "nested_select": bool(index % 5 == 0),
                    "schema_table_bucket": "1-5",
                },
            }
        )

    first = stratified_sample(
        records,
        size=24,
        seed="fixed",
        max_database_share=0.25,
    )
    second = stratified_sample(
        records,
        size=24,
        seed="fixed",
        max_database_share=0.25,
    )

    assert [row["case_id"] for row in first] == [
        row["case_id"] for row in second
    ]
    counts: dict[str, int] = {}
    for row in first:
        db_id = str(row["database_id"])
        counts[db_id] = counts.get(db_id, 0) + 1
    assert max(counts.values()) <= 6


def test_builder_execution_json_and_validation_redaction(
    demo_db: Path,
    tmp_path: Path,
) -> None:
    sql = "SELECT COUNT(*) AS customer_count FROM customers"
    cases_path = tmp_path / "cases.json"
    output_path = tmp_path / "train.jsonl"
    tools_path = tmp_path / "tools.json"
    cases_path.write_text(
        json.dumps(
            {
                "dataset_name": "bird_fixture",
                "cases": [
                    {
                        "case_id": "bird.case",
                        "question": "How many customers are there?",
                        "sql": sql,
                        "inspect_tables": ["customers"],
                        "tool_turns": [
                            {"name": "list_tables"},
                            {
                                "name": "inspect_schema",
                                "arguments": {
                                    "table_names": ["customers"]
                                },
                            },
                            {"name": "validate_sql"},
                            {"name": "execute_sql"},
                        ],
                        "answer": {"kind": "execution_json"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    samples = build_dataset(
        database=demo_db,
        cases_path=cases_path,
        output=output_path,
        tools_output=tools_path,
        split="train",
    )

    sample = samples[0]
    assert json.loads(sample["messages"][-1]["content"]) == {
        "columns": ["customer_count"],
        "rows": [[10]],
    }
    validation_message = next(
        message
        for message in sample["messages"]
        if message.get("role") == "tool"
        and message.get("name") == "validate_sql"
    )
    validation_payload = json.loads(validation_message["content"])
    assert "normalized_sql" not in validation_payload["output"]["safety_check"]
    assert sql not in validation_message["content"]


def test_low_entropy_evidence_is_not_a_false_positive() -> None:
    sample = {
        "messages": [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "list",
                        "function": {
                            "name": "list_tables",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "list",
                "name": "list_tables",
                "content": '{"truncated":false}',
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "execute",
                        "function": {
                            "name": "execute_sql",
                            "arguments": '{"sql":"SELECT 1"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "execute",
                "name": "execute_sql",
                "content": '{"success":true}',
            },
            {"role": "assistant", "content": "done"},
        ]
    }

    failures = _audit_messages(
        sample=sample,
        question="Question",
        evidence="false",
        gold_sql="SELECT 1",
    )

    assert "evidence_in_tool_observation" not in failures


def test_builder_cli_accepts_frozen_dev_monitoring_contract() -> None:
    args = build_orchestration_parser().parse_args(
        [
            "build",
            "--annotations",
            "train.json",
            "--pool-manifest",
            "monitoring.json",
            "--database-archive",
            "databases.zip",
            "--temporary-root",
            "tmp",
            "--output",
            "dev.jsonl",
            "--tools-output",
            "dev_tools.json",
            "--audit-output",
            "dev_audit.json",
            "--model",
            "models/Qwen3-8B",
            "--split",
            "dev",
            "--expected-cases",
            "128",
            "--dataset-name",
            "bird_sft_val_monitoring_v1",
        ]
    )

    assert args.model == Path("models/Qwen3-8B")
    assert args.split == "dev"
    assert args.expected_cases == 128
    assert args.dataset_name == "bird_sft_val_monitoring_v1"
