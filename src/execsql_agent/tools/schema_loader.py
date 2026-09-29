"""Read structured SQLite schema metadata without modifying the database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

from execsql_agent.models import (
    ColumnSchema,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)


def _readonly_uri(database_path: Path) -> str:
    return f"{database_path.resolve().as_uri()}?mode=ro"


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class SchemaLoader:
    """Load SQLite tables, columns, primary keys, and foreign keys."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def list_tables(self) -> list[str]:
        """Return deterministic user-defined table names without loading columns."""

        if not self.database_path.is_file():
            raise FileNotFoundError(f"Database does not exist: {self.database_path}")

        with sqlite3.connect(_readonly_uri(self.database_path), uri=True) as connection:
            connection.execute("PRAGMA query_only = ON")
            rows = cast(
                list[tuple[str]],
                connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall(),
            )
        return [name for (name,) in rows]

    def load(self) -> DatabaseSchema:
        """Return deterministic structured and text schema representations."""

        if not self.database_path.is_file():
            raise FileNotFoundError(f"Database does not exist: {self.database_path}")

        with sqlite3.connect(_readonly_uri(self.database_path), uri=True) as connection:
            connection.execute("PRAGMA query_only = ON")
            table_rows = cast(
                list[tuple[str]],
                connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall(),
            )
            tables = [self._load_table(connection, name) for (name,) in table_rows]

        return DatabaseSchema(
            database_id=self.database_path.stem,
            tables=tables,
            summary_text=self._render_summary(tables),
        )

    def _load_table(self, connection: sqlite3.Connection, table_name: str) -> TableSchema:
        quoted_name = _quote_identifier(table_name)
        column_rows = cast(
            list[tuple[int, str, str, int, str | None, int]],
            connection.execute(f"PRAGMA table_info({quoted_name})").fetchall(),
        )
        foreign_key_rows = cast(
            list[tuple[int, int, str, str, str | None, str, str, str]],
            connection.execute(f"PRAGMA foreign_key_list({quoted_name})").fetchall(),
        )

        columns: list[ColumnSchema] = []
        ordered_primary_keys: list[tuple[int, str]] = []
        for _, name, data_type, not_null, default_value, primary_key_position in column_rows:
            is_primary_key = primary_key_position > 0
            columns.append(
                ColumnSchema(
                    name=name,
                    data_type=data_type or "ANY",
                    nullable=not bool(not_null) and not is_primary_key,
                    default_value=default_value,
                    primary_key=is_primary_key,
                )
            )
            if is_primary_key:
                ordered_primary_keys.append((primary_key_position, name))

        foreign_keys = [
            ForeignKeySchema(
                source_table=table_name,
                source_column=source_column,
                target_table=target_table,
                target_column=target_column,
                on_update=on_update,
                on_delete=on_delete,
            )
            for (
                _,
                _,
                target_table,
                source_column,
                target_column,
                on_update,
                on_delete,
                _,
            ) in foreign_key_rows
        ]
        return TableSchema(
            name=table_name,
            columns=columns,
            primary_keys=[name for _, name in sorted(ordered_primary_keys)],
            foreign_keys=foreign_keys,
        )

    @staticmethod
    def _render_summary(tables: list[TableSchema]) -> str:
        lines: list[str] = []
        for table in tables:
            lines.append(f"TABLE {table.name}")
            for column in table.columns:
                attributes: list[str] = []
                if column.primary_key:
                    attributes.append("PRIMARY KEY")
                if not column.nullable:
                    attributes.append("NOT NULL")
                suffix = f" {' '.join(attributes)}" if attributes else ""
                lines.append(f"  - {column.name} {column.data_type}{suffix}")
            for foreign_key in table.foreign_keys:
                target = foreign_key.target_table
                if foreign_key.target_column is not None:
                    target += f"({foreign_key.target_column})"
                lines.append(
                    "  - FOREIGN KEY "
                    f"{foreign_key.source_column} REFERENCES "
                    f"{target}"
                )
        return "\n".join(lines)
