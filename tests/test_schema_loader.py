"""Tests for read-only SQLite schema introspection."""

from pathlib import Path

import pytest

from execsql_agent.tools.schema_loader import SchemaLoader


def test_loads_tables_columns_and_primary_keys(demo_db: Path) -> None:
    schema = SchemaLoader(demo_db).load()
    tables = {table.name: table for table in schema.tables}

    assert schema.database_id == "demo"
    assert list(tables) == ["customers", "order_items", "orders", "products"]
    assert tables["customers"].primary_keys == ["customer_id"]
    assert [column.name for column in tables["customers"].columns] == [
        "customer_id",
        "customer_name",
        "city",
        "signup_date",
    ]
    assert tables["customers"].columns[0].nullable is False


def test_loads_foreign_keys(demo_db: Path) -> None:
    schema = SchemaLoader(demo_db).load()
    tables = {table.name: table for table in schema.tables}

    order_foreign_key = tables["orders"].foreign_keys[0]
    assert order_foreign_key.source_column == "customer_id"
    assert order_foreign_key.target_table == "customers"
    assert order_foreign_key.target_column == "customer_id"

    item_targets = {
        (foreign_key.source_column, foreign_key.target_table)
        for foreign_key in tables["order_items"].foreign_keys
    }
    assert item_targets == {("order_id", "orders"), ("product_id", "products")}


def test_renders_deterministic_model_summary(demo_db: Path) -> None:
    summary = SchemaLoader(demo_db).load().summary_text

    assert "TABLE customers" in summary
    assert "customer_id INTEGER PRIMARY KEY NOT NULL" in summary
    assert "FOREIGN KEY customer_id REFERENCES customers(customer_id)" in summary
    assert summary.index("TABLE customers") < summary.index("TABLE orders")


def test_missing_database_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Database does not exist"):
        SchemaLoader(tmp_path / "missing.db").load()
