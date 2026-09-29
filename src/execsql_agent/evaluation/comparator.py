"""Result-based correctness comparison for real SQLite outputs."""

from __future__ import annotations

from math import isclose

from execsql_agent.models import ExecutionResult, ExpectedResult


def _cell_matches(actual: object, expected: object, tolerance: float) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if (
        isinstance(actual, (int, float))
        and isinstance(expected, (int, float))
    ):
        return isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
    if isinstance(actual, (int, float)) or isinstance(expected, (int, float)):
        return False
    if isinstance(actual, str) or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (bytes, bytearray, memoryview)) or isinstance(
        expected, (bytes, bytearray, memoryview)
    ):
        return isinstance(actual, (bytes, bytearray, memoryview)) and isinstance(
            expected, (bytes, bytearray, memoryview)
        ) and bytes(actual) == bytes(expected)
    return type(actual) is type(expected) and actual == expected


def _row_matches(actual: list[object], expected: list[object], tolerance: float) -> bool:
    return len(actual) == len(expected) and all(
        _cell_matches(left, right, tolerance)
        for left, right in zip(actual, expected, strict=True)
    )


def _unordered_rows_match(
    actual: list[list[object]], expected: list[list[object]], tolerance: float
) -> bool:
    """Compare multisets with greedy one-to-one matching and duplicate preservation."""

    if len(actual) != len(expected):
        return False
    unmatched = list(expected)
    for actual_row in actual:
        for index, expected_row in enumerate(unmatched):
            if _row_matches(actual_row, expected_row, tolerance):
                unmatched.pop(index)
                break
        else:
            return False
    return not unmatched


def compare_execution_result(
    actual: ExecutionResult | None, expected: ExpectedResult
) -> bool | None:
    """Return correctness, or ``None`` when no successful real result is available."""

    if actual is None or not actual.execution_success:
        return None
    if actual.truncated:
        return False
    if len(actual.columns) != len(expected.columns):
        return False
    if expected.strict_columns and actual.columns != expected.columns:
        return False
    if expected.ordered:
        if len(actual.rows) != len(expected.rows):
            return False
        return all(
            _row_matches(left, right, expected.numeric_tolerance)
            for left, right in zip(actual.rows, expected.rows, strict=True)
        )
    return _unordered_rows_match(
        actual.rows, expected.rows, expected.numeric_tolerance
    )


def compare_bird_execution_result(
    actual: ExecutionResult | None, expected: ExpectedResult
) -> bool:
    """Match official BIRD EX set semantics over complete SQLite result rows."""

    if actual is None or not actual.execution_success or actual.truncated:
        return False
    predicted_rows = {tuple(row) for row in actual.rows}
    gold_rows = {tuple(row) for row in expected.rows}
    return predicted_rows == gold_rows
