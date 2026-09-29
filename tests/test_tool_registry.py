"""Strict allowlist and real SQLite tests for ToolRegistry."""

import json
import sqlite3
from pathlib import Path

from execsql_agent.models import ToolCallRequest
from execsql_agent.tools.registry import ToolRegistry


def test_exports_exactly_four_strict_json_schemas(demo_db: Path) -> None:
    definitions = ToolRegistry(demo_db).definitions

    assert [definition.name for definition in definitions] == [
        "inspect_schema",
        "validate_sql",
        "execute_sql",
        "list_tables",
    ]
    assert all(definition.parameters["additionalProperties"] is False for definition in definitions)
    execute_schema = definitions[2].parameters
    assert execute_schema["required"] == ["sql"]
    list_tables = definitions[3]
    assert list_tables.parameters["properties"] == {}
    assert list_tables.description == (
        "List the available tables in the current database. Use this when the "
        "relevant table names are unknown before inspecting detailed schema."
    )


def test_list_tables_returns_sorted_user_tables_and_serializes(tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE zeta (id INTEGER)")
        connection.execute("CREATE TABLE alpha (id INTEGER)")
        connection.execute(
            "CREATE TABLE sequenced (id INTEGER PRIMARY KEY AUTOINCREMENT)"
        )

    registry = ToolRegistry(database)
    call = ToolCallRequest(id="catalog", name="list_tables", arguments={})
    first = registry.dispatch(call)[1]
    second = registry.dispatch(call)[1]

    assert first.success is True
    assert first.executed is True
    assert first.output == {
        "tables": ["alpha", "sequenced", "zeta"],
        "count": 3,
    }
    assert second.output == first.output
    assert "sqlite_sequence" not in first.output["tables"]
    serialized = json.loads(first.model_dump_json())
    assert serialized["output"] == first.output


def test_list_tables_supports_single_table_database(tmp_path: Path) -> None:
    database = tmp_path / "single.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE only_table (id INTEGER)")

    _, result = ToolRegistry(database).dispatch(
        ToolCallRequest(id="catalog", name="list_tables", arguments={})
    )

    assert result.output == {"tables": ["only_table"], "count": 1}


def test_unknown_tool_is_never_executed(demo_db: Path) -> None:
    validation, result = ToolRegistry(demo_db).dispatch(
        ToolCallRequest(id="bad", name="drop_database", arguments={})
    )

    assert validation.known_tool is False
    assert result.success is False
    assert result.executed is False
    assert result.error_code == "unknown_tool"


def test_invalid_json_arguments_are_never_executed(demo_db: Path) -> None:
    validation, result = ToolRegistry(demo_db).dispatch(
        ToolCallRequest(
            id="bad-json",
            name="execute_sql",
            arguments=None,
            raw_arguments="{bad",
        )
    )

    assert validation.arguments_valid is False
    assert result.executed is False
    assert result.error_code == "invalid_json_arguments"


def test_missing_type_and_extra_arguments_are_rejected(demo_db: Path) -> None:
    registry = ToolRegistry(demo_db)
    calls = [
        ToolCallRequest(id="missing", name="execute_sql", arguments={}),
        ToolCallRequest(id="type", name="execute_sql", arguments={"sql": 42}),
        ToolCallRequest(
            id="extra",
            name="execute_sql",
            arguments={"sql": "SELECT 1", "unexpected": True},
        ),
    ]

    results = [registry.dispatch(call)[1] for call in calls]

    assert all(result.executed is False for result in results)
    assert all(result.error_code == "invalid_arguments" for result in results)


def test_inspect_validate_and_execute_use_real_database(demo_db: Path) -> None:
    registry = ToolRegistry(demo_db)
    _, schema_result = registry.dispatch(
        ToolCallRequest(
            id="schema",
            name="inspect_schema",
            arguments={"table_names": ["customers"]},
        )
    )
    _, validation_result = registry.dispatch(
        ToolCallRequest(
            id="validate",
            name="validate_sql",
            arguments={"sql": "SELECT customer_id FROM customers"},
        )
    )
    _, execution_result = registry.dispatch(
        ToolCallRequest(
            id="execute",
            name="execute_sql",
            arguments={"sql": "SELECT COUNT(*) AS count FROM customers"},
        )
    )

    assert schema_result.success is True
    assert schema_result.output is not None
    assert schema_result.output["tables"][0]["name"] == "customers"
    assert "schema_text" not in schema_result.output
    assert validation_result.success is True
    assert execution_result.success is True
    typed_execution = registry.execution_result(execution_result)
    assert typed_execution is not None
    assert typed_execution.rows == [[10]]


def test_inspect_schema_preserves_implicit_foreign_key_target_column(
    tmp_path: Path,
) -> None:
    database = tmp_path / "implicit-foreign-key.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE child (parent_id INTEGER REFERENCES parent)"
        )
        raw_foreign_key = connection.execute(
            "PRAGMA foreign_key_list(child)"
        ).fetchone()
    assert raw_foreign_key is not None
    assert raw_foreign_key[4] is None

    _, result = ToolRegistry(database).dispatch(
        ToolCallRequest(
            id="implicit-fk",
            name="inspect_schema",
            arguments={"table_names": ["child"]},
        )
    )

    assert result.success is True
    assert result.output is not None
    tables = result.output["tables"]
    assert isinstance(tables, list)
    foreign_key = tables[0]["foreign_keys"][0]
    assert foreign_key["target_table"] == "parent"
    assert foreign_key["target_column"] is None
    assert '"target_column": null' in json.dumps(result.output)
    assert "schema_text" not in result.output


def test_targeted_inspect_preserves_selected_columns_and_primary_key(
    demo_db: Path,
) -> None:
    _, result = ToolRegistry(demo_db).dispatch(
        ToolCallRequest(
            id="one-table",
            name="inspect_schema",
            arguments={"table_names": ["customers"]},
        )
    )

    assert result.success is True
    assert result.output is not None
    assert "schema_text" not in result.output
    assert [table["name"] for table in result.output["tables"]] == ["customers"]
    customer = result.output["tables"][0]
    assert customer["primary_keys"] == ["customer_id"]
    assert [column["name"] for column in customer["columns"]] == [
        "customer_id",
        "customer_name",
        "city",
        "signup_date",
    ]


def test_targeted_inspect_preserves_join_foreign_key_and_serializes(
    demo_db: Path,
) -> None:
    _, result = ToolRegistry(demo_db).dispatch(
        ToolCallRequest(
            id="related-tables",
            name="inspect_schema",
            arguments={"table_names": ["customers", "orders"]},
        )
    )

    assert result.success is True
    assert result.output is not None
    assert "schema_text" not in result.output
    tables = {table["name"]: table for table in result.output["tables"]}
    assert list(tables) == ["customers", "orders"]
    assert tables["customers"]["primary_keys"] == ["customer_id"]
    assert tables["orders"]["foreign_keys"] == [
        {
            "source_table": "orders",
            "source_column": "customer_id",
            "target_table": "customers",
            "target_column": "customer_id",
            "on_update": "NO ACTION",
            "on_delete": "NO ACTION",
        }
    ]
    serialized = json.loads(result.model_dump_json())
    assert serialized["output"] == result.output


def test_unfiltered_inspect_variants_keep_full_schema_observation(
    demo_db: Path,
) -> None:
    registry = ToolRegistry(demo_db)
    arguments = [{}, {"table_names": None}, {"table_names": []}]

    outputs = [
        registry.dispatch(
            ToolCallRequest(id=f"all-{index}", name="inspect_schema", arguments=value)
        )[1].output
        for index, value in enumerate(arguments)
    ]

    assert all(output is not None for output in outputs)
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0] is not None
    assert [table["name"] for table in outputs[0]["tables"]] == [
        "customers",
        "order_items",
        "orders",
        "products",
    ]
    assert "TABLE customers" in outputs[0]["schema_text"]
    assert "TABLE products" in outputs[0]["schema_text"]


def test_unknown_table_and_partial_match_behavior_are_unchanged(
    demo_db: Path,
) -> None:
    registry = ToolRegistry(demo_db)
    _, unknown = registry.dispatch(
        ToolCallRequest(
            id="unknown",
            name="inspect_schema",
            arguments={"table_names": ["missing_table"]},
        )
    )
    _, partial = registry.dispatch(
        ToolCallRequest(
            id="partial",
            name="inspect_schema",
            arguments={"table_names": ["customers", "missing_table"]},
        )
    )

    assert unknown.success is False
    assert unknown.executed is True
    assert unknown.error_code == "unknown_table"
    assert unknown.error_message == "Unknown table(s): missing_table"
    assert unknown.output == {"database_id": "demo", "tables": []}
    assert partial.success is False
    assert partial.executed is True
    assert partial.error_code == "unknown_table"
    assert partial.error_message == "Unknown table(s): missing_table"
    assert partial.output is not None
    assert [table["name"] for table in partial.output["tables"]] == ["customers"]
    assert "schema_text" not in partial.output


def test_unsafe_sql_never_reaches_sqlite(demo_db: Path) -> None:
    registry = ToolRegistry(demo_db)

    _, validation = registry.dispatch(
        ToolCallRequest(
            id="validate", name="validate_sql", arguments={"sql": "DELETE FROM customers"}
        )
    )
    _, execution = registry.dispatch(
        ToolCallRequest(
            id="execute", name="execute_sql", arguments={"sql": "DELETE FROM customers"}
        )
    )

    assert validation.error_code == "unsafe_sql"
    assert validation.executed is False
    assert execution.error_code == "unsafe_sql"
    assert execution.executed is False
