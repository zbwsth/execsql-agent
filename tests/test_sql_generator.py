"""Tests for initial and repair SQL generation."""

import json
from pathlib import Path

from execsql_agent.generation.sql_generator import SQLGenerator
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import (
    DiagnosisSource,
    ErrorDiagnosis,
    ErrorType,
    GenerationMode,
    LLMResponse,
    ResponseMode,
)
from execsql_agent.tools.schema_loader import SchemaLoader


def _response(sql: str, reason: str) -> LLMResponse:
    return LLMResponse(
        final_answer=json.dumps(
            {
                "sql": sql,
                "reason": reason,
                "referenced_tables": ["customers"],
                "referenced_columns": ["customer_id"],
            }
        ),
        response_mode=ResponseMode.PLAIN_FINAL,
    )


def test_initial_generation_uses_schema_and_validates_output(demo_db: Path) -> None:
    fake = FakeLLMClient([_response("SELECT customer_id FROM customers", "initial")])
    generator = SQLGenerator(fake)

    result = generator.generate(
        question="列出客户编号",
        schema=SchemaLoader(demo_db).load(),
        mode=GenerationMode.INITIAL_GENERATION,
    )

    assert result.sql == "SELECT customer_id FROM customers"
    request_payload = json.loads(fake.requests[0].messages[1].content)
    assert request_payload["mode"] == "initial_generation"
    assert "TABLE customers" in request_payload["schema"]


def test_repair_generation_includes_error_and_instruction(demo_db: Path) -> None:
    fake = FakeLLMClient([_response("SELECT customer_id FROM customers", "repair")])
    generator = SQLGenerator(fake)
    diagnosis = ErrorDiagnosis(
        error_type=ErrorType.MISSING_TABLE,
        cause="customer does not exist",
        repair_instruction="Use customers.",
        related_tables=["customers"],
        source=DiagnosisSource.RULE,
    )

    result = generator.generate(
        question="列出客户编号",
        schema=SchemaLoader(demo_db).load(),
        mode=GenerationMode.REPAIR,
        history_sql=["SELECT customer_id FROM customer"],
        previous_error="no such table: customer",
        diagnosis=diagnosis,
    )

    assert result.reason == "repair"
    request_payload = json.loads(fake.requests[0].messages[1].content)
    assert request_payload["mode"] == "repair"
    assert request_payload["previous_error"] == "no such table: customer"
    assert request_payload["diagnosis"]["repair_instruction"] == "Use customers."
