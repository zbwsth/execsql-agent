"""End-to-end tests for the fixed Pipeline Agent using real SQLite."""

import json
from pathlib import Path

from execsql_agent.agents.pipeline import PipelineAgent
from execsql_agent.generation.sql_generator import SQLGenerator
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import (
    ErrorType,
    GenerationMode,
    LLMError,
    LLMResponse,
    PipelineTerminationReason,
    ResponseMode,
)


def _generation(sql: str, reason: str = "test") -> LLMResponse:
    return LLMResponse(
        final_answer=json.dumps(
            {
                "sql": sql,
                "reason": reason,
                "referenced_tables": [],
                "referenced_columns": [],
            }
        ),
        response_mode=ResponseMode.PLAIN_FINAL,
    )


def _agent(
    demo_db: Path,
    responses: list[LLMResponse | LLMError],
    **kwargs: object,
) -> PipelineAgent:
    return PipelineAgent(demo_db, SQLGenerator(FakeLLMClient(responses)), **kwargs)


def test_pipeline_initial_execution_success(demo_db: Path) -> None:
    agent = _agent(demo_db, [_generation("SELECT COUNT(*) AS count FROM customers")])

    result = agent.run("客户有多少名？")

    assert result.termination_reason is PipelineTerminationReason.COMPLETED
    assert result.protocol_completed is True
    assert result.execution_success is True
    assert result.answer_grounded is True
    assert result.result_correct is None
    assert result.execution_result is not None
    assert result.execution_result.rows == [[10]]
    assert result.total_steps == 1


def test_pipeline_repairs_wrong_table_name(demo_db: Path) -> None:
    agent = _agent(
        demo_db,
        [
            _generation("SELECT customer_name FROM customer", "wrong table"),
            _generation(
                "SELECT customer_name FROM customers ORDER BY customer_id LIMIT 2",
                "fixed table",
            ),
        ],
    )

    result = agent.run("列出前两名客户")

    assert result.termination_reason is PipelineTerminationReason.COMPLETED
    assert result.total_steps == 2
    assert result.steps[0].diagnosis is not None
    assert result.steps[0].diagnosis.error_type is ErrorType.MISSING_TABLE
    assert result.steps[1].mode is GenerationMode.REPAIR
    assert result.execution_result is not None
    assert result.execution_result.rows == [["张伟"], ["王芳"]]


def test_pipeline_stops_on_repeated_normalized_sql(demo_db: Path) -> None:
    agent = _agent(
        demo_db,
        [
            _generation("SELECT * FROM customer"),
            _generation(" select  *  from  CUSTOMER ;"),
        ],
    )

    result = agent.run("列出客户")

    assert result.termination_reason is PipelineTerminationReason.REPEATED_SQL
    assert result.total_steps == 2
    assert result.steps[1].execution is None
    assert result.steps[1].diagnosis is not None
    assert result.steps[1].diagnosis.error_type is ErrorType.REPEATED_SQL


def test_pipeline_repairs_first_unsafe_candidate(demo_db: Path) -> None:
    agent = _agent(
        demo_db,
        [
            _generation("DELETE FROM customers", "unsafe"),
            _generation("SELECT COUNT(*) FROM customers", "safe repair"),
        ],
    )

    result = agent.run("客户数量")

    assert result.termination_reason is PipelineTerminationReason.COMPLETED
    assert result.steps[0].execution is None
    assert result.steps[0].diagnosis is not None
    assert result.steps[0].diagnosis.error_type is ErrorType.UNSAFE_SQL
    assert result.execution_result is not None
    assert result.execution_result.rows == [[10]]


def test_pipeline_stops_after_second_unsafe_candidate(demo_db: Path) -> None:
    agent = _agent(
        demo_db,
        [_generation("DELETE FROM customers"), _generation("DROP TABLE customers")],
    )

    result = agent.run("删除客户")

    assert result.termination_reason is PipelineTerminationReason.UNSAFE_SQL
    assert result.execution_success is False
    assert all(step.execution is None for step in result.steps)


def test_pipeline_stops_at_max_steps(demo_db: Path) -> None:
    agent = _agent(
        demo_db,
        [
            _generation("SELECT * FROM missing_one"),
            _generation("SELECT * FROM missing_two"),
            _generation("SELECT * FROM missing_three"),
        ],
    )

    result = agent.run("查询不存在的表")

    assert result.termination_reason is PipelineTerminationReason.MAX_STEPS_REACHED
    assert result.total_steps == 3
    assert result.execution_success is False


def test_pipeline_surfaces_model_error(demo_db: Path) -> None:
    agent = _agent(
        demo_db,
        [LLMError(code="model_error", message="model unavailable", retryable=True)],
    )

    result = agent.run("客户数量")

    assert result.termination_reason is PipelineTerminationReason.MODEL_ERROR
    assert result.protocol_completed is False
    assert result.steps[0].llm_error is not None
    assert result.steps[0].llm_error.code == "model_error"


def test_pipeline_empty_result_is_success_by_default(demo_db: Path) -> None:
    agent = _agent(
        demo_db,
        [_generation("SELECT customer_id FROM customers WHERE customer_id < 0")],
    )

    result = agent.run("不存在的客户")

    assert result.termination_reason is PipelineTerminationReason.COMPLETED
    assert result.execution_success is True
    assert result.execution_result is not None
    assert result.execution_result.rows == []
    assert result.execution_result.returned_row_count == 0
