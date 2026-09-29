"""Build execution-verified Function Calling SFT data from explicit case specs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from execsql_agent.models import ExecutionResult, ToolCallRequest, ToolCallResult
from execsql_agent.tools.registry import ToolRegistry

HARNESS_V2_TOOL_NAMES = frozenset(
    {
        "list_tables",
        "inspect_schema",
        "validate_sql",
        "execute_sql",
    }
)
DEFAULT_TOOL_SEQUENCE = (
    "list_tables",
    "inspect_schema",
    "validate_sql",
    "execute_sql",
)
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")


def _load_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return raw


def _normalized_question(question: str) -> str:
    return "".join(question.split()).casefold()


def _validate_tool_turns(
    case_id: str, case: dict[str, Any]
) -> None:
    raw_turns = case.get("tool_turns")
    if raw_turns is None:
        return
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError(f"{case_id}: tool_turns must be a non-empty array")
    normalized: list[dict[str, object]] = []
    for index, raw_turn in enumerate(raw_turns, start=1):
        if not isinstance(raw_turn, dict):
            raise ValueError(
                f"{case_id}: tool turn {index} must be an object"
            )
        name = raw_turn.get("name")
        arguments = raw_turn.get("arguments", {})
        if not isinstance(name, str) or name not in HARNESS_V2_TOOL_NAMES:
            raise ValueError(
                f"{case_id}: tool turn {index} has unknown tool {name!r}"
            )
        if not isinstance(arguments, dict):
            raise ValueError(
                f"{case_id}: tool turn {index} arguments must be an object"
            )
        normalized.append(
            {"name": name, "arguments": dict(arguments)}
        )
    if not any(turn["name"] == "execute_sql" for turn in normalized):
        raise ValueError(
            f"{case_id}: tool_turns must include execute_sql"
        )
    case["tool_turns"] = normalized


def _load_cases(path: Path) -> tuple[str, list[dict[str, Any]]]:
    raw = _load_json_object(path)
    dataset_name = raw.get("dataset_name")
    raw_cases = raw.get("cases")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("Case spec requires a non-empty dataset_name")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Case spec requires a non-empty cases array")

    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Case {index} must be a JSON object")
        case = dict(raw_case)
        case_id = case.get("case_id")
        question = case.get("question")
        sql = case.get("sql")
        inspect_tables = case.get("inspect_tables")
        answer = case.get("answer")
        if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id):
            raise ValueError(f"Invalid case_id: {case_id!r}")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{case_id}: question must be non-empty")
        normalized = _normalized_question(question)
        if normalized in seen_questions:
            raise ValueError(f"{case_id}: duplicate normalized question")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"{case_id}: sql must be non-empty")
        if not isinstance(inspect_tables, list) or not all(
            isinstance(table, str) and table for table in inspect_tables
        ):
            raise ValueError(f"{case_id}: inspect_tables must be a string array")
        if not isinstance(answer, dict):
            raise ValueError(f"{case_id}: answer must be an object")
        _validate_tool_turns(case_id, case)
        case.setdefault("template_family", "general")
        case.setdefault("difficulty", "unknown")
        case.setdefault("parameter_signature", case_id)
        seen_ids.add(case_id)
        seen_questions.add(normalized)
        cases.append(case)
    return dataset_name, cases


def _tool_schemas(registry: ToolRegistry) -> list[dict[str, object]]:
    definitions = registry.definitions
    names = {definition.name for definition in definitions}
    expected = set(HARNESS_V2_TOOL_NAMES)
    if names != expected:
        raise ValueError(
            "ToolRegistry definitions differ from Harness v2: "
            f"expected={sorted(expected)}, actual={sorted(names)}"
        )
    return [
        {"type": "function", "function": definition.model_dump(mode="json")}
        for definition in definitions
    ]


def _dispatch(registry: ToolRegistry, call: ToolCallRequest) -> ToolCallResult:
    if call.name not in HARNESS_V2_TOOL_NAMES:
        raise ValueError(f"Forbidden tool: {call.name}")
    validation, result = registry.dispatch(call)
    if not validation.valid:
        raise ValueError(
            f"{call.id}: invalid {call.name} call: "
            f"{validation.error_code}: {validation.error_message}"
        )
    if not result.success:
        raise ValueError(
            f"{call.id}: {call.name} failed: "
            f"{result.error_code}: {result.error_message}"
        )
    return result


def _assistant_tool_message(call: ToolCallRequest) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            }
        ],
    }


def _tool_result_message(
    call: ToolCallRequest, result: ToolCallResult
) -> dict[str, object]:
    payload = result.model_dump(mode="json")
    if call.name == "validate_sql":
        output = payload.get("output")
        if isinstance(output, dict):
            safety = output.get("safety_check")
            if isinstance(safety, dict):
                safety = dict(safety)
                safety.pop("normalized_sql", None)
                output = dict(output)
                output["safety_check"] = safety
                payload["output"] = output
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _row_mapping(
    execution: ExecutionResult, row: list[object]
) -> dict[str, object]:
    if len(execution.columns) != len(row):
        raise ValueError("Execution columns and row width differ")
    return dict(zip(execution.columns, row, strict=True))


def _render_answer(
    answer: dict[str, Any], execution: ExecutionResult
) -> str:
    kind = answer.get("kind")
    if kind == "execution_json":
        return json.dumps(
            {
                "columns": execution.columns,
                "rows": execution.rows,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    template = answer.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("answer.template must be non-empty")
    if kind == "scalar":
        if len(execution.rows) != 1:
            raise ValueError("A scalar answer requires exactly one result row")
        return template.format_map(
            _row_mapping(execution, execution.rows[0])
        )
    if kind == "rows":
        row_template = answer.get("row_template")
        if not isinstance(row_template, str) or not row_template.strip():
            raise ValueError("A rows answer requires row_template")
        rendered_rows: list[str] = []
        for rank, row in enumerate(execution.rows, start=1):
            values = _row_mapping(execution, row)
            values["rank"] = rank
            rendered_rows.append(row_template.format_map(values))
        return template.format(rows="; ".join(rendered_rows))
    raise ValueError(f"Unknown answer kind: {kind!r}")


def _build_tool_calls(
    case: dict[str, Any], sql: str
) -> list[ToolCallRequest]:
    case_id = str(case["case_id"])
    raw_turns = case.get("tool_turns")
    if raw_turns is None:
        raw_turns = [
            {"name": "list_tables", "arguments": {}},
            {"name": "inspect_schema", "arguments": {}},
            {"name": "validate_sql", "arguments": {}},
            {"name": "execute_sql", "arguments": {}},
        ]
    assert isinstance(raw_turns, list)

    calls: list[ToolCallRequest] = []
    for index, raw_turn in enumerate(raw_turns, start=1):
        assert isinstance(raw_turn, dict)
        name = str(raw_turn["name"])
        raw_arguments = raw_turn.get("arguments", {})
        assert isinstance(raw_arguments, dict)
        arguments = dict(raw_arguments)
        if name == "inspect_schema" and "table_names" not in arguments:
            arguments["table_names"] = case["inspect_tables"]
        if name in {"validate_sql", "execute_sql"} and "sql" not in arguments:
            arguments["sql"] = sql
        calls.append(
            ToolCallRequest(
                id=f"{case_id}_{index:02d}_{name}",
                name=name,
                arguments=arguments,
            )
        )
    if not any(call.name == "execute_sql" for call in calls):
        raise ValueError(f"{case_id}: trajectory has no execute_sql turn")
    return calls


def _build_sample(
    *,
    dataset_name: str,
    split: str,
    case: dict[str, Any],
    registry: ToolRegistry,
    tools: list[dict[str, object]],
) -> dict[str, object]:
    case_id = str(case["case_id"])
    question = str(case["question"])
    sql = str(case["sql"])
    safety = registry.validator.validate(
        sql, database_path=registry.database_path
    )
    if not safety.safe or safety.syntax_valid is not True:
        raise ValueError(
            f"{case_id}: ground-truth SQL is not safe and compilable: "
            f"{safety.reason or safety.validation_error}"
        )

    calls = _build_tool_calls(case, sql)

    messages: list[dict[str, object]] = [
        {"role": "user", "content": question}
    ]
    execute_result: ToolCallResult | None = None
    for call in calls:
        result = _dispatch(registry, call)
        messages.append(_assistant_tool_message(call))
        messages.append(_tool_result_message(call, result))
        if call.name == "execute_sql":
            execute_result = result

    if execute_result is None:
        raise ValueError(f"{case_id}: missing execute_sql result")
    execution = registry.execution_result(execute_result)
    if (
        execution is None
        or not execution.executed
        or not execution.execution_success
    ):
        raise ValueError(f"{case_id}: execute_sql did not succeed")
    if execution.truncated:
        raise ValueError(f"{case_id}: final result is truncated")

    messages.append(
        {
            "role": "assistant",
            "content": _render_answer(case["answer"], execution),
        }
    )
    return {
        "messages": messages,
        "tools": tools,
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "case_id": case_id,
            "template_family": case["template_family"],
            "difficulty": case["difficulty"],
            "parameter_signature": case["parameter_signature"],
            "tool_sequence": [call.name for call in calls],
            "gold_sql": sql,
            "execution_result": execution.model_dump(mode="json"),
        },
    }


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _serialize_sample(sample: dict[str, object]) -> str:
    return json.dumps(
        sample, ensure_ascii=False, separators=(",", ":")
    )


def _atomic_write_jsonl(
    path: Path, samples: Sequence[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(_serialize_sample(sample))
            handle.write("\n")
    temporary.replace(path)


def _validate_jsonl(path: Path, expected_ids: set[str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(expected_ids):
        raise ValueError(
            f"Expected {len(expected_ids)} JSONL lines, found {len(lines)}"
        )
    actual_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(
                f"JSONL line {line_number} is not an object"
            )
        metadata = parsed.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                f"JSONL line {line_number} has no metadata object"
            )
        case_id = metadata.get("case_id")
        if not isinstance(case_id, str) or case_id not in expected_ids:
            raise ValueError(
                f"JSONL line {line_number} has an unexpected case id"
            )
        if case_id in actual_ids:
            raise ValueError(f"Duplicate output case id: {case_id}")
        actual_ids.add(case_id)
        execution = metadata.get("execution_result")
        if not isinstance(execution, dict):
            raise ValueError(
                f"JSONL line {line_number} has no execution_result"
            )
        if (
            execution.get("execution_success") is not True
            or execution.get("truncated") is not False
        ):
            raise ValueError(
                f"JSONL line {line_number} has an invalid execution result"
            )
        messages = parsed.get("messages")
        if not isinstance(messages, list):
            raise ValueError(
                f"JSONL line {line_number} has no messages array"
            )
        tool_names = [
            call["function"]["name"]
            for message in messages
            if isinstance(message, dict)
            for call in message.get("tool_calls", [])
            if isinstance(call, dict)
            and isinstance(call.get("function"), dict)
        ]
        if (
            not tool_names
            or "execute_sql" not in tool_names
            or any(
                name not in HARNESS_V2_TOOL_NAMES
                for name in tool_names
            )
        ):
            raise ValueError(
                f"JSONL line {line_number} has an invalid tool sequence"
            )
        if metadata.get("tool_sequence") != tool_names:
            raise ValueError(
                f"JSONL line {line_number} tool sequence metadata differs"
            )


def _print_statistics(
    samples: Sequence[dict[str, object]]
) -> None:
    family_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    for sample in samples:
        metadata = sample["metadata"]
        assert isinstance(metadata, dict)
        family_counts[str(metadata["template_family"])] += 1
        difficulty_counts[str(metadata["difficulty"])] += 1
    print(f"samples={len(samples)}")
    print(
        "template_family_counts="
        + json.dumps(
            dict(sorted(family_counts.items())), ensure_ascii=False
        )
    )
    print(
        "difficulty_counts="
        + json.dumps(
            dict(sorted(difficulty_counts.items())), ensure_ascii=False
        )
    )


def build_dataset(
    *,
    database: Path,
    cases_path: Path,
    output: Path,
    tools_output: Path,
    split: str,
) -> list[dict[str, object]]:
    if split not in {"train", "dev"}:
        raise ValueError("split must be 'train' or 'dev'")
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")
    dataset_name, cases = _load_cases(cases_path)
    registry = ToolRegistry(database)
    tools = _tool_schemas(registry)
    samples = [
        _build_sample(
            dataset_name=dataset_name,
            split=split,
            case=case,
            registry=registry,
            tools=tools,
        )
        for case in cases
    ]
    _atomic_write_json(tools_output, tools)
    _atomic_write_jsonl(output, samples)
    _validate_jsonl(
        output,
        {str(case["case_id"]) for case in cases},
    )
    _print_statistics(samples)
    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build execution-verified Function Calling SFT data from "
            "an explicit, database-agnostic case specification."
        )
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["train", "dev"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tools-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = build_dataset(
        database=args.database,
        cases_path=args.cases,
        output=args.output,
        tools_output=args.tools_output,
        split=args.split,
    )
    print(
        f"Generated {len(samples)} {args.split} samples: {args.output}"
    )
    print(f"Exported ToolRegistry schemas: {args.tools_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
