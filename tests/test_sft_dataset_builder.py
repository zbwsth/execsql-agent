"""Generic execution-verified SFT builder tests."""

from __future__ import annotations

import json
from pathlib import Path

from training.build_sft_dataset import (
    DEFAULT_TOOL_SEQUENCE,
    HARNESS_V2_TOOL_NAMES,
    build_dataset,
)


def test_builder_uses_explicit_sqlite_fixture_and_harness_v2_tools(
    demo_db: Path,
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.json"
    output_path = tmp_path / "train.jsonl"
    tools_path = tmp_path / "tools.json"
    cases_path.write_text(
        json.dumps(
            {
                "dataset_name": "generic_sqlite_fixture",
                "cases": [
                    {
                        "case_id": "customers.count",
                        "question": "How many customers are there?",
                        "sql": (
                            "SELECT COUNT(*) AS customer_count "
                            "FROM customers"
                        ),
                        "inspect_tables": ["customers"],
                        "answer": {
                            "kind": "scalar",
                            "template": "There are {customer_count} customers.",
                        },
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

    assert len(samples) == 1
    sample = samples[0]
    metadata = sample["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["dataset_name"] == "generic_sqlite_fixture"
    assert metadata["split"] == "train"

    messages = sample["messages"]
    assert isinstance(messages, list)
    tool_names = [
        call["function"]["name"]
        for message in messages
        if isinstance(message, dict)
        for call in message.get("tool_calls", [])
    ]
    assert tuple(tool_names) == DEFAULT_TOOL_SEQUENCE
    assert messages[-1]["content"] == "There are 10 customers."

    exported_tools = json.loads(tools_path.read_text(encoding="utf-8"))
    assert {
        definition["function"]["name"] for definition in exported_tools
    } == HARNESS_V2_TOOL_NAMES
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1


def test_case_spec_can_choose_different_legal_tool_turns(
    demo_db: Path,
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "policies.json"
    output_path = tmp_path / "dev.jsonl"
    tools_path = tmp_path / "tools.json"
    cases_path.write_text(
        json.dumps(
            {
                "dataset_name": "trajectory_policy_fixture",
                "cases": [
                    {
                        "case_id": "customers.no_discovery",
                        "question": "Count customers without table discovery.",
                        "sql": (
                            "SELECT COUNT(*) AS customer_count "
                            "FROM customers"
                        ),
                        "inspect_tables": ["customers"],
                        "tool_turns": [
                            {"name": "inspect_schema"},
                            {"name": "validate_sql"},
                            {"name": "execute_sql"},
                        ],
                        "answer": {
                            "kind": "scalar",
                            "template": "{customer_count} customers",
                        },
                    },
                    {
                        "case_id": "orders.two_inspections",
                        "question": "Count orders after inspecting two tables.",
                        "sql": "SELECT COUNT(*) AS order_count FROM orders",
                        "inspect_tables": ["orders"],
                        "tool_turns": [
                            {"name": "list_tables"},
                            {
                                "name": "inspect_schema",
                                "arguments": {
                                    "table_names": ["customers"]
                                },
                            },
                            {
                                "name": "inspect_schema",
                                "arguments": {"table_names": ["orders"]},
                            },
                            {"name": "validate_sql"},
                            {"name": "execute_sql"},
                        ],
                        "answer": {
                            "kind": "scalar",
                            "template": "{order_count} orders",
                        },
                    },
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
        split="dev",
    )

    sequences = {
        sample["metadata"]["case_id"]: tuple(
            sample["metadata"]["tool_sequence"]
        )
        for sample in samples
    }
    assert sequences == {
        "customers.no_discovery": (
            "inspect_schema",
            "validate_sql",
            "execute_sql",
        ),
        "orders.two_inspections": (
            "list_tables",
            "inspect_schema",
            "inspect_schema",
            "validate_sql",
            "execute_sql",
        ),
    }
