"""BIRD Mini-Dev adapter, multi-database scoring, and isolation tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from execsql_agent.evaluation.bird import (
    BirdGoldCacheEntry,
    load_bird_dataset,
    make_bird_database_resolver,
    preflight_bird_dataset,
    resolve_bird_database_path,
)
from execsql_agent.evaluation.comparator import compare_bird_execution_result
from execsql_agent.evaluation.evaluator import Evaluator
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import (
    BirdPredictionTimeoutPolicy,
    EvaluationCase,
    ExecutionError,
    ExecutionResult,
    ExpectedResult,
    FailureKind,
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
)
from execsql_agent.tools.sql_executor import SQLExecutor


def _create_database(root: Path, db_id: str, label: str) -> Path:
    directory = root / db_id
    directory.mkdir(parents=True)
    database = directory / f"{db_id}.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE numbers (value REAL NOT NULL)")
        connection.executemany(
            "INSERT INTO numbers (value) VALUES (?)",
            [(float(value),) for value in range(1, 151)],
        )
        connection.execute("CREATE TABLE metadata (label TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata (label) VALUES (?)", (label,))
        connection.execute("CREATE TABLE private_gold (secret TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO private_gold (secret) VALUES ('GOLD_ROW_SENTINEL')"
        )
    return database


@pytest.fixture
def bird_root(tmp_path: Path) -> Path:
    root = tmp_path / "dev_databases"
    root.mkdir()
    _create_database(root, "db_alpha", "alpha")
    _create_database(root, "db_beta", "beta")
    return root


def _record(
    question_id: int,
    *,
    db_id: str = "db_alpha",
    sql: str = "SELECT label FROM metadata",
    difficulty: str = "simple",
    evidence: str = "EVIDENCE_SENTINEL",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "db_id": db_id,
        "question": f"Question {question_id}?",
        "SQL": sql,
        "difficulty": difficulty,
        "evidence": evidence,
    }


def _write_json(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _execution(rows: list[list[object]], *, success: bool = True) -> ExecutionResult:
    return ExecutionResult(
        executed=True,
        execution_success=success,
        columns=["value"],
        rows=rows if success else [],
        returned_row_count=len(rows) if success else 0,
        truncated=False,
        duration_ms=0,
    )


def _sql_client(sql: str) -> FakeLLMClient:
    return FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="execute", name="execute_sql", arguments={"sql": sql}
                    )
                ],
                response_mode=ResponseMode.NATIVE_TOOL_CALLS,
            ),
            LLMResponse(
                final_answer="done", response_mode=ResponseMode.PLAIN_FINAL
            ),
        ]
    )


def _direct_answer_client() -> FakeLLMClient:
    return FakeLLMClient(
        [
            LLMResponse(final_answer="done", response_mode=ResponseMode.PLAIN_FINAL),
            LLMResponse(
                final_answer="still unresolved",
                response_mode=ResponseMode.PLAIN_FINAL,
            ),
        ]
    )


def _inspect_then_answer_client() -> FakeLLMClient:
    return FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="inspect", name="inspect_schema", arguments={}
                    )
                ],
                response_mode=ResponseMode.NATIVE_TOOL_CALLS,
            ),
            LLMResponse(
                final_answer="done", response_mode=ResponseMode.PLAIN_FINAL
            ),
            LLMResponse(
                final_answer="still unresolved",
                response_mode=ResponseMode.PLAIN_FINAL,
            ),
        ]
    )


def _request_text(client: FakeLLMClient) -> str:
    return "\n".join(
        message.model_dump_json()
        for request in client.requests
        for message in request.messages
    )


def test_json_loader_maps_fields_and_builds_full_expected_result(
    bird_root: Path, tmp_path: Path
) -> None:
    dataset_path = _write_json(
        tmp_path / "mini_dev.json",
        [_record(1471, sql="SELECT value FROM numbers")],
    )

    dataset = load_bird_dataset(dataset_path, bird_root, expected_count=1)
    case = dataset.cases[0]

    assert dataset.dataset_name == "bird_mini_dev_sqlite"
    assert case.id == "bird_1471"
    assert case.database_id == "db_alpha"
    assert case.gold_sql == "SELECT value FROM numbers"
    assert case.difficulty == "simple"
    assert case.evidence == "EVIDENCE_SENTINEL"
    assert case.comparison_mode == "bird_set"
    assert case.expected_result.ordered is False
    assert len(case.expected_result.rows) == 150


def test_jsonl_loader_supports_multiple_databases(
    bird_root: Path, tmp_path: Path
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "mini_dev.jsonl",
        [_record(1), _record(2, db_id="db_beta", difficulty="moderate")],
    )

    dataset = load_bird_dataset(dataset_path, bird_root, expected_count=2)

    assert [case.id for case in dataset.cases] == ["bird_1", "bird_2"]
    assert [case.database_id for case in dataset.cases] == ["db_alpha", "db_beta"]


def test_gold_cache_writes_reads_and_preserves_full_rows(
    bird_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = _write_json(
        tmp_path / "cache.json",
        [_record(1, sql="SELECT value FROM numbers ORDER BY value")],
    )
    cache_path = tmp_path / "cache" / "gold.jsonl"

    first = preflight_bird_dataset(
        dataset_path,
        bird_root,
        cache_path,
        expected_count=1,
    )

    assert first.cache_stats.hits == 0
    assert first.cache_stats.misses == 1
    assert len(first.cache_entries[0].rows) == 150
    assert first.cache_entries[0].row_count == 150
    cached = BirdGoldCacheEntry.model_validate_json(
        cache_path.read_text(encoding="utf-8").strip()
    )
    assert cached.rows == first.cache_entries[0].rows
    assert cached.database_fingerprint.sha256

    def unexpected_execute_full(
        _self: SQLExecutor, _sql: str, *, timeout_seconds: float = 30.0
    ) -> ExecutionResult:
        raise AssertionError(f"cache hit executed SQL with timeout {timeout_seconds}")

    monkeypatch.setattr(SQLExecutor, "execute_full", unexpected_execute_full)
    resumed = preflight_bird_dataset(
        dataset_path,
        bird_root,
        cache_path,
        expected_count=1,
    )

    assert resumed.cache_stats.hits == 1
    assert resumed.cache_stats.misses == 0
    assert len(resumed.dataset.cases[0].expected_result.rows) == 150


def test_frozen_evaluation_requires_valid_cache_without_rewriting_it(
    bird_root: Path, tmp_path: Path
) -> None:
    dataset_path = _write_json(tmp_path / "frozen.json", [_record(1)])
    cache_path = tmp_path / "gold.jsonl"

    with pytest.raises(ValueError, match="cache entry is missing"):
        load_bird_dataset(
            dataset_path,
            bird_root,
            expected_count=1,
            gold_cache_path=cache_path,
            gold_cache_read_only=True,
        )
    assert not cache_path.exists()

    preflight_bird_dataset(
        dataset_path, bird_root, cache_path, expected_count=1
    )
    original = cache_path.read_bytes()
    frozen = load_bird_dataset(
        dataset_path,
        bird_root,
        expected_count=1,
        gold_cache_path=cache_path,
        gold_cache_read_only=True,
    )

    assert frozen.cases[0].gold_runtime_ms is not None
    assert cache_path.read_bytes() == original


def test_gold_cache_invalidates_changed_sql(
    bird_root: Path, tmp_path: Path
) -> None:
    dataset_path = _write_json(tmp_path / "sql-change.json", [_record(1)])
    cache_path = tmp_path / "gold.jsonl"
    first = preflight_bird_dataset(
        dataset_path, bird_root, cache_path, expected_count=1
    )
    old_hash = first.cache_entries[0].gold_sql_sha256

    _write_json(
        dataset_path,
        [_record(1, sql="SELECT COUNT(*) FROM metadata")],
    )
    second = preflight_bird_dataset(
        dataset_path, bird_root, cache_path, expected_count=1
    )

    assert second.cache_stats.hits == 0
    assert second.cache_stats.misses == 1
    assert second.cache_stats.invalidated == 1
    assert second.cache_entries[0].gold_sql_sha256 != old_hash
    assert second.cache_entries[0].rows == [[1]]


def test_gold_cache_invalidates_changed_database_fingerprint(
    bird_root: Path, tmp_path: Path
) -> None:
    dataset_path = _write_json(tmp_path / "db-change.json", [_record(1)])
    cache_path = tmp_path / "gold.jsonl"
    first = preflight_bird_dataset(
        dataset_path, bird_root, cache_path, expected_count=1
    )
    old_fingerprint = first.cache_entries[0].database_fingerprint

    database = resolve_bird_database_path(bird_root, "db_alpha")
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO metadata VALUES ('changed')")
    second = preflight_bird_dataset(
        dataset_path, bird_root, cache_path, expected_count=1
    )

    assert second.cache_stats.invalidated == 1
    assert second.cache_entries[0].database_fingerprint != old_fingerprint
    assert second.cache_entries[0].row_count == 2


def test_gold_cache_persists_each_success_and_resumes_after_failure(
    bird_root: Path, tmp_path: Path
) -> None:
    dataset_path = _write_json(
        tmp_path / "partial.json",
        [_record(1), _record(2, sql="SELECT missing FROM metadata")],
    )
    cache_path = tmp_path / "gold.jsonl"

    with pytest.raises(ValueError, match="failed validation"):
        preflight_bird_dataset(
            dataset_path,
            bird_root,
            cache_path,
            expected_count=2,
        )

    lines = cache_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert BirdGoldCacheEntry.model_validate_json(lines[0]).case_id == "bird_1"
    assert not list(cache_path.parent.glob(f".{cache_path.name}.*.tmp"))

    _write_json(dataset_path, [_record(1), _record(2, db_id="db_beta")])
    resumed = preflight_bird_dataset(
        dataset_path,
        bird_root,
        cache_path,
        expected_count=2,
    )

    assert resumed.cache_stats.hits == 1
    assert resumed.cache_stats.misses == 1
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 2


def test_loader_rejects_invalid_difficulty(bird_root: Path, tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "invalid.json", [_record(1, difficulty="hard")]
    )

    with pytest.raises(ValueError, match="Invalid BIRD difficulty"):
        load_bird_dataset(path, bird_root, expected_count=1)


def test_loader_rejects_crud_gold_sql(bird_root: Path, tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "crud.json", [_record(1, sql="DELETE FROM numbers")]
    )

    with pytest.raises(ValueError, match="read-only SELECT/WITH"):
        load_bird_dataset(path, bird_root, expected_count=1)


def test_loader_requires_expected_case_count(bird_root: Path, tmp_path: Path) -> None:
    path = _write_json(tmp_path / "count.json", [_record(1)])

    with pytest.raises(ValueError, match="Expected 500 BIRD cases"):
        load_bird_dataset(path, bird_root)


def test_database_path_is_exact_and_missing_database_is_hard_failure(
    bird_root: Path, tmp_path: Path
) -> None:
    resolved = resolve_bird_database_path(bird_root, "db_alpha")
    assert resolved == (bird_root / "db_alpha" / "db_alpha.sqlite").resolve()

    missing = _write_json(
        tmp_path / "missing.json", [_record(1, db_id="missing")]
    )
    with pytest.raises(ValueError, match="database does not exist"):
        load_bird_dataset(missing, bird_root, expected_count=1)
    with pytest.raises(ValueError, match="Invalid BIRD db_id"):
        resolve_bird_database_path(bird_root, "../db_alpha")


def test_full_executor_does_not_change_bounded_agent_behavior(bird_root: Path) -> None:
    database = resolve_bird_database_path(bird_root, "db_alpha")
    executor = SQLExecutor(database)

    bounded = executor.execute("SELECT value FROM numbers ORDER BY value")
    complete = executor.execute_full("SELECT value FROM numbers ORDER BY value")

    assert len(bounded.rows) == 100
    assert bounded.truncated is True
    assert len(complete.rows) == 150
    assert complete.truncated is False


def test_bird_set_semantics_are_distinct_from_tolerant_multiset() -> None:
    expected = ExpectedResult(columns=["ignored"], rows=[[1.0], [2.0]], ordered=False)

    assert compare_bird_execution_result(_execution([[2.0], [1.0]]), expected)
    assert compare_bird_execution_result(_execution([[2.0], [1.0], [1.0]]), expected)
    assert not compare_bird_execution_result(_execution([[1.0000005], [2.0]]), expected)
    assert not compare_bird_execution_result(_execution([[1.0], [3.0]]), expected)
    assert not compare_bird_execution_result(None, expected)


def test_multi_database_routing_and_more_than_100_rows_score_correctly(
    bird_root: Path, tmp_path: Path
) -> None:
    path = _write_json(
        tmp_path / "routing.json",
        [
            _record(1, db_id="db_alpha", sql="SELECT value FROM numbers"),
            _record(2, db_id="db_beta", sql="SELECT label FROM metadata"),
        ],
    )
    dataset = load_bird_dataset(path, bird_root, expected_count=2)

    def factory(case: EvaluationCase, _mode: str) -> FakeLLMClient:
        sql = (
            "SELECT value FROM numbers"
            if case.database_id == "db_alpha"
            else "SELECT label AS predicted_label FROM metadata"
        )
        return _sql_client(sql)

    report = Evaluator(
        None,
        factory,
        database_resolver=make_bird_database_resolver(bird_root),
    ).evaluate(
        dataset.cases,
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
        seed=0,
    )

    assert report.database_id == "multiple"
    assert report.database_ids == ["db_alpha", "db_beta"]
    assert all(case.result_correct is True for case in report.cases)
    alpha = next(case for case in report.cases if case.database_id == "db_alpha")
    assert alpha.actual_result is not None
    assert len(alpha.actual_result.rows) == 150
    assert alpha.actual_result.truncated is False
    bird_ex = report.mode_metrics["function-calling"].bird_ex
    assert bird_ex is not None
    assert (bird_ex.overall.numerator, bird_ex.overall.denominator) == (2, 2)
    runtime = report.mode_metrics["function-calling"].prediction_sql_runtime
    assert runtime is not None
    assert runtime.measured_count == 2
    assert runtime.timeout_count == 0
    stable = report.mode_metrics["function-calling"].stable_bird_ex
    ex_at_30s = report.mode_metrics["function-calling"].ex_at_30s
    assert stable is not None and stable.overall.value == 1
    assert ex_at_30s is not None and ex_at_30s.overall.value == 1


def test_prediction_timeout_policy_and_ex_at_30s_use_one_execution(
    bird_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = BirdPredictionTimeoutPolicy()
    assert policy.timeout_seconds(1_000) == 30.0
    assert policy.timeout_seconds(70_000) == 110.0
    assert policy.timeout_seconds(1_000_000) == 900.0
    dataset_path = _write_json(tmp_path / "runtime.json", [_record(1)])
    case = load_bird_dataset(
        dataset_path, bird_root, expected_count=1
    ).cases[0].model_copy(update={"gold_runtime_ms": 70_000.0})
    scorer_calls = 0

    def delayed_correct_result(
        _self: SQLExecutor, _sql: str, *, timeout_seconds: float = 30.0
    ) -> ExecutionResult:
        nonlocal scorer_calls
        scorer_calls += 1
        assert timeout_seconds == 110.0
        return ExecutionResult(
            executed=True,
            execution_success=True,
            columns=["label"],
            rows=[["alpha"]],
            returned_row_count=1,
            truncated=False,
            duration_ms=31_000.0,
        )

    monkeypatch.setattr(SQLExecutor, "execute_full", delayed_correct_result)
    report = Evaluator(
        None,
        lambda _case, _mode: _sql_client("SELECT label FROM metadata"),
        database_resolver=make_bird_database_resolver(bird_root),
        prediction_timeout_policy=policy,
    ).evaluate(
        [case],
        dataset_name="bird-runtime",
        agent_mode="function-calling",
    )
    evaluated = report.cases[0]

    assert scorer_calls == 1
    assert evaluated.prediction_timeout_budget_seconds == 110.0
    assert evaluated.prediction_runtime_ms == 31_000.0
    assert evaluated.prediction_timeout is False
    assert evaluated.stable_bird_ex is True
    assert evaluated.ex_at_30s is False
    metrics = report.mode_metrics["function-calling"]
    assert metrics.stable_bird_ex is not None
    assert metrics.stable_bird_ex.overall.value == 1
    assert metrics.ex_at_30s is not None
    assert metrics.ex_at_30s.overall.value == 0
    assert metrics.prediction_sql_runtime is not None
    assert metrics.prediction_sql_runtime.median_ms == 31_000.0


def test_prediction_timeout_is_reported_and_scores_zero(
    bird_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = _write_json(tmp_path / "prediction-timeout.json", [_record(1)])
    case = load_bird_dataset(
        dataset_path, bird_root, expected_count=1
    ).cases[0]

    def timed_out(
        _self: SQLExecutor, _sql: str, *, timeout_seconds: float = 30.0
    ) -> ExecutionResult:
        return ExecutionResult(
            executed=True,
            execution_success=False,
            error=ExecutionError(
                message=f"query timed out after {timeout_seconds:.1f} seconds"
            ),
            duration_ms=timeout_seconds * 1000,
        )

    monkeypatch.setattr(SQLExecutor, "execute_full", timed_out)
    report = Evaluator(
        None,
        lambda _case, _mode: _sql_client("SELECT label FROM metadata"),
        database_resolver=make_bird_database_resolver(bird_root),
        prediction_timeout_policy=BirdPredictionTimeoutPolicy(),
    ).evaluate(
        [case], dataset_name="bird-timeout", agent_mode="function-calling"
    )

    evaluated = report.cases[0]
    assert evaluated.prediction_timeout is True
    assert evaluated.stable_bird_ex is False
    assert evaluated.ex_at_30s is False
    assert evaluated.failure_kind is FailureKind.EXECUTION_ERROR
    runtime = report.mode_metrics["function-calling"].prediction_sql_runtime
    assert runtime is not None
    assert runtime.timeout_count == 1
    assert runtime.timeout_rate.value == 1


def test_gold_and_prediction_timeouts_are_independently_configurable(
    bird_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = _write_json(tmp_path / "timeouts.json", [_record(1)])
    cache_path = tmp_path / "gold.jsonl"
    observed_timeouts: list[float] = []
    original_execute_full = SQLExecutor.execute_full

    def capture_timeout(
        self: SQLExecutor, sql: str, *, timeout_seconds: float = 30.0
    ) -> ExecutionResult:
        observed_timeouts.append(timeout_seconds)
        return original_execute_full(self, sql, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(SQLExecutor, "execute_full", capture_timeout)
    preflight = preflight_bird_dataset(
        dataset_path,
        bird_root,
        cache_path,
        expected_count=1,
        gold_timeout_seconds=7.0,
    )
    Evaluator(
        None,
        lambda _case, _mode: _sql_client("SELECT label FROM metadata"),
        database_resolver=make_bird_database_resolver(bird_root),
        prediction_timeout_seconds=0.5,
    ).evaluate(
        preflight.dataset.cases,
        dataset_name=preflight.dataset.dataset_name,
        agent_mode="function-calling",
    )

    assert observed_timeouts == [7.0, 0.5]


def test_bird_failures_count_as_zero_without_overwriting_failure_kind(
    bird_root: Path, tmp_path: Path
) -> None:
    path = _write_json(
        tmp_path / "failures.json",
        [
            _record(1, difficulty="simple"),
            _record(2, difficulty="moderate"),
            _record(3, db_id="db_beta", difficulty="challenging"),
        ],
    )
    dataset = load_bird_dataset(path, bird_root, expected_count=3)

    def factory(case: EvaluationCase, _mode: str) -> FakeLLMClient:
        if case.difficulty == "simple":
            return _sql_client("SELECT label AS answer FROM metadata")
        if case.difficulty == "moderate":
            return _sql_client("SELECT missing_column FROM metadata")
        return _direct_answer_client()

    report = Evaluator(
        None,
        factory,
        database_resolver=make_bird_database_resolver(bird_root),
    ).evaluate(
        dataset.cases,
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
    )
    by_difficulty = {case.difficulty: case for case in report.cases}

    assert by_difficulty["simple"].result_correct is True
    assert by_difficulty["moderate"].result_correct is False
    assert by_difficulty["moderate"].failure_kind is FailureKind.EXECUTION_ERROR
    assert by_difficulty["challenging"].final_sql is None
    assert by_difficulty["challenging"].result_correct is False
    assert by_difficulty["challenging"].protocol_completed is False
    assert by_difficulty["challenging"].termination_reason == "unresolved_completion"
    assert (
        by_difficulty["challenging"].failure_kind
        is FailureKind.UNRESOLVED_COMPLETION
    )

    bird_ex = report.mode_metrics["function-calling"].bird_ex
    assert bird_ex is not None
    assert (bird_ex.simple.numerator, bird_ex.simple.denominator) == (1, 1)
    assert (bird_ex.moderate.numerator, bird_ex.moderate.denominator) == (0, 1)
    assert (bird_ex.challenging.numerator, bird_ex.challenging.denominator) == (0, 1)
    assert (bird_ex.overall.numerator, bird_ex.overall.denominator) == (1, 3)


def test_wrong_bird_result_is_semantic_mismatch(
    bird_root: Path, tmp_path: Path
) -> None:
    path = _write_json(tmp_path / "wrong.json", [_record(1)])
    dataset = load_bird_dataset(path, bird_root, expected_count=1)

    report = Evaluator(
        None,
        lambda _case, _mode: _sql_client("SELECT 'wrong'"),
        database_resolver=make_bird_database_resolver(bird_root),
    ).evaluate(
        dataset.cases,
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
    )

    assert report.cases[0].result_correct is False
    assert report.cases[0].failure_kind is FailureKind.SEMANTIC_MISMATCH


@pytest.mark.parametrize("use_evidence", [False, True])
def test_evidence_and_scorer_metadata_are_isolated_from_all_llm_requests(
    bird_root: Path,
    tmp_path: Path,
    use_evidence: bool,
) -> None:
    gold_sql = "SELECT secret FROM private_gold"
    path = _write_json(
        tmp_path / f"leak-{use_evidence}.json",
        [_record(1, sql=gold_sql, evidence="EVIDENCE_SENTINEL")],
    )
    dataset = load_bird_dataset(path, bird_root, expected_count=1)
    clients: list[FakeLLMClient] = []

    def factory(_case: EvaluationCase, _mode: str) -> FakeLLMClient:
        client = _inspect_then_answer_client()
        clients.append(client)
        return client

    report = Evaluator(
        None,
        factory,
        database_resolver=make_bird_database_resolver(bird_root),
        use_evidence=use_evidence,
    ).evaluate(
        dataset.cases,
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
    )

    assert report.oracle_evidence is use_evidence
    assert len(clients[0].requests) == 3
    request_text = _request_text(clients[0])
    assert gold_sql not in request_text
    assert "GOLD_ROW_SENTINEL" not in request_text
    assert "expected_result" not in request_text.casefold()
    assert "bird_set" not in request_text
    if use_evidence:
        assert "EVIDENCE_SENTINEL" in request_text
    else:
        assert "EVIDENCE_SENTINEL" not in request_text


def test_strict_loader_rejects_unknown_fields(bird_root: Path, tmp_path: Path) -> None:
    record = _record(1)
    record["unexpected"] = True
    path = _write_json(tmp_path / "extra.json", [record])

    with pytest.raises(ValidationError):
        load_bird_dataset(path, bird_root, expected_count=1)
