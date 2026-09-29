"""Static safety checks and optional SQLite compile validation."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from execsql_agent.models import SafetyCheckResult

_ALLOWED_START_KEYWORDS = {"SELECT", "WITH"}
_BLOCKED_KEYWORDS = {
    "ALTER",
    "ATTACH",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "UPDATE",
    "VACUUM",
}


@dataclass(frozen=True)
class _ScanResult:
    """SQL fragments collected without interpreting quoted content."""

    statements: list[str]
    keywords: list[str]
    malformed_reason: str | None = None


def _scan_sql(sql: str) -> _ScanResult:
    statements: list[str] = []
    keywords: list[str] = []
    current: list[str] = []
    word: list[str] = []
    state = "normal"
    index = 0

    def flush_word() -> None:
        if word:
            keywords.append("".join(word).upper())
            word.clear()

    def finish_statement() -> None:
        fragment = "".join(current).strip()
        if fragment:
            statements.append(fragment)
        current.clear()

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if state == "line_comment":
            if char in "\r\n":
                state = "normal"
                current.append(" ")
            index += 1
            continue

        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                current.append(" ")
                index += 2
            else:
                index += 1
            continue

        if state in {"single", "double", "backtick"}:
            current.append(char)
            delimiter = {"single": "'", "double": '"', "backtick": "`"}[state]
            if char == delimiter:
                if next_char == delimiter:
                    current.append(next_char)
                    index += 2
                    continue
                state = "normal"
            index += 1
            continue

        if state == "bracket":
            current.append(char)
            if char == "]":
                if next_char == "]":
                    current.append(next_char)
                    index += 2
                    continue
                state = "normal"
            index += 1
            continue

        if char == "-" and next_char == "-":
            flush_word()
            state = "line_comment"
            index += 2
            continue
        if char == "/" and next_char == "*":
            flush_word()
            state = "block_comment"
            index += 2
            continue
        if char in {"'", '"', "`", "["}:
            flush_word()
            state = {"'": "single", '"': "double", "`": "backtick", "[": "bracket"}[char]
            current.append(char)
            index += 1
            continue
        if char == ";":
            flush_word()
            finish_statement()
            index += 1
            continue

        current.append(char)
        if char.isalnum() or char == "_":
            word.append(char)
        else:
            flush_word()
        index += 1

    flush_word()
    if state in {"single", "double", "backtick", "bracket", "block_comment"}:
        return _ScanResult(statements, keywords, f"Unterminated SQL {state} section.")
    finish_statement()
    return _ScanResult(statements, keywords)


def _readonly_uri(database_path: Path) -> str:
    return f"{database_path.resolve().as_uri()}?mode=ro"


def _normalize_statement(statement: str) -> str:
    """Normalize unquoted tokens while preserving all quoted content exactly."""

    tokens: list[tuple[str, bool]] = []
    index = 0
    while index < len(statement):
        char = statement[index]
        if char.isspace():
            index += 1
            continue

        if char in {"'", '"', "`", "["}:
            start = index
            delimiter = "]" if char == "[" else char
            index += 1
            while index < len(statement):
                current = statement[index]
                if current == delimiter:
                    if index + 1 < len(statement) and statement[index + 1] == delimiter:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            tokens.append((statement[start:index], True))
            continue

        if char.isalnum() or char in {"_", "$"}:
            start = index
            index += 1
            while index < len(statement):
                current = statement[index]
                if not (current.isalnum() or current in {"_", "$"}):
                    break
                index += 1
            tokens.append((statement[start:index].lower(), True))
            continue

        tokens.append((char.lower(), False))
        index += 1

    normalized: list[str] = []
    previous_is_atom = False
    for token, is_atom in tokens:
        if normalized and previous_is_atom and is_atom:
            normalized.append(" ")
        normalized.append(token)
        previous_is_atom = is_atom
    return "".join(normalized)


class SQLValidator:
    """Validate that SQL is one read-only SELECT statement."""

    def validate(
        self,
        sql: str,
        *,
        database_path: str | Path | None = None,
    ) -> SafetyCheckResult:
        """Check safety and optionally ask SQLite to compile the query plan."""

        scan = _scan_sql(sql)
        normalized = " ".join(_normalize_statement(statement) for statement in scan.statements)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        if scan.malformed_reason is not None:
            return SafetyCheckResult(
                safe=False,
                normalized_sql=normalized,
                reason=scan.malformed_reason,
                blocked_operation="MALFORMED_SQL",
            )
        if not scan.statements:
            return SafetyCheckResult(
                safe=False,
                normalized_sql="",
                reason="SQL must not be empty.",
                blocked_operation="EMPTY_SQL",
            )
        if len(scan.statements) != 1:
            return SafetyCheckResult(
                safe=False,
                normalized_sql=normalized,
                reason="Only one SQL statement is allowed.",
                blocked_operation="MULTI_STATEMENT",
            )

        blocked = next((word for word in scan.keywords if word in _BLOCKED_KEYWORDS), None)
        if blocked is not None:
            return SafetyCheckResult(
                safe=False,
                normalized_sql=normalized,
                reason=f"Operation {blocked} is not allowed.",
                blocked_operation=blocked,
            )

        first_keyword = scan.keywords[0] if scan.keywords else ""
        if first_keyword not in _ALLOWED_START_KEYWORDS:
            return SafetyCheckResult(
                safe=False,
                normalized_sql=normalized,
                reason="Only SELECT statements and SELECT CTEs are allowed.",
                blocked_operation=first_keyword or "UNKNOWN",
            )

        result = SafetyCheckResult(safe=True, normalized_sql=normalized)
        if database_path is None:
            return result

        path = Path(database_path)
        if not path.is_file():
            return result.model_copy(
                update={
                    "syntax_valid": False,
                    "validation_error": f"Database does not exist: {path}",
                }
            )

        try:
            with sqlite3.connect(_readonly_uri(path), uri=True) as connection:
                connection.execute("PRAGMA query_only = ON")
                connection.execute(f"EXPLAIN QUERY PLAN {scan.statements[0]}")
        except sqlite3.Error as error:
            return result.model_copy(
                update={"syntax_valid": False, "validation_error": str(error)}
            )
        return result.model_copy(update={"syntax_valid": True})
