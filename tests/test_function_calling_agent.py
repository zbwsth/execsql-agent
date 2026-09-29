"""End-to-end FunctionCallingAgent tests with real SQLite tools."""

import json
from pathlib import Path

import httpx

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.llm.openai_compatible import (
    OpenAICompatibleLLMClient,
    OpenAICompatibleSettings,
)
from execsql_agent.models import (
    CompletionKind,
    FunctionCallingTerminationReason,
    LLMError,
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
)


def _calls(*calls: ToolCallRequest, fallback: bool = False) -> LLMResponse:
    return LLMResponse(
        tool_calls=list(calls),
        response_mode=(
            ResponseMode.JSON_FALLBACK if fallback else ResponseMode.NATIVE_TOOL_CALLS
        ),
    )


def _final(text: str = "完成", *, fallback: bool = False) -> LLMResponse:
    return LLMResponse(
        final_answer=text,
        response_mode=(
            ResponseMode.JSON_FALLBACK if fallback else ResponseMode.PLAIN_FINAL
        ),
    )


def _tool(call_id: str, name: str, arguments: dict[str, object] | None) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def test_inspect_execute_final_and_multiturn_history(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(_tool("schema", "inspect_schema", {})),
            _calls(
                _tool(
                    "execute",
                    "execute_sql",
                    {"sql": "SELECT COUNT(*) AS count FROM customers"},
                )
            ),
            _final("共有 10 名客户。"),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("有多少客户？")

    assert result.termination_reason is FunctionCallingTerminationReason.COMPLETED
    assert result.completion_kind is CompletionKind.TOOL_GROUNDED_ANSWER
    assert result.protocol_completed is True
    assert result.execution_success is True
    assert result.answer_grounded is True
    assert result.result_correct is None
    assert result.execution_result is not None
    assert result.execution_result.rows == [[10]]
    assert all(
        "The task has not yet produced" not in (message.content or "")
        for request in fake.requests
        for message in request.messages
    )

    second_request_messages = fake.requests[1].messages
    assistant = second_request_messages[-2]
    observation = second_request_messages[-1]
    assert assistant.role == "assistant"
    assert assistant.tool_calls[0].id == "schema"
    assert observation.role == "tool"
    assert observation.tool_call_id == "schema"


def test_list_tables_is_dynamically_dispatched_and_observed(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(_tool("catalog", "list_tables", {})),
            _calls(_tool("execute", "execute_sql", {"sql": "SELECT 1"})),
            _final("The available tables were discovered."),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("Discover the database tables.")

    assert result.termination_reason is FunctionCallingTerminationReason.COMPLETED
    assert result.steps[0].tool_results[0].tool_name == "list_tables"
    assert result.steps[0].tool_results[0].output == {
        "tables": ["customers", "order_items", "orders", "products"],
        "count": 4,
    }
    observation = fake.requests[1].messages[-1]
    assert observation.role == "tool"
    assert observation.tool_call_id == "catalog"
    assert json.loads(observation.content or "{}")["output"] == {
        "tables": ["customers", "order_items", "orders", "products"],
        "count": 4,
    }


def test_completion_guard_allows_dynamic_tool_recovery_after_inspect(
    demo_db: Path,
) -> None:
    fake = FakeLLMClient(
        [
            _calls(
                _tool(
                    "schema",
                    "inspect_schema",
                    {"table_names": ["customers"]},
                )
            ),
            _final("I should query the customer table next."),
            _calls(
                _tool(
                    "execute",
                    "execute_sql",
                    {"sql": "SELECT COUNT(*) AS count FROM customers"},
                )
            ),
            _final("There are 10 customers."),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("How many customers are there?")

    guard_text = (
        "The task has not yet produced or successfully executed SQL. "
        "If the database can answer the question, continue using the available tools. "
        "If it cannot be completed, explicitly explain why."
    )
    guarded_request = fake.requests[2]
    assert guarded_request.messages[0].role == "system"
    assert guard_text in (guarded_request.messages[0].content or "")
    assert [
        message.content
        for message in guarded_request.messages
        if message.role == "user"
    ] == ["How many customers are there?"]
    assert all(
        guard_text not in (message.content or "")
        for request in fake.requests
        for message in request.messages
        if message.role == "user"
    )
    assert guard_text not in (fake.requests[-1].messages[0].content or "")
    assert result.steps[1].completion_guard_triggered is True
    assert result.steps[1].completion_guard_instruction == guard_text
    assert result.termination_reason is FunctionCallingTerminationReason.COMPLETED
    assert result.protocol_completed is True
    assert result.execution_success is True
    assert result.total_llm_turns == 4


def test_completion_guard_is_used_once_then_unresolved(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(
                _tool(
                    "schema",
                    "inspect_schema",
                    {"table_names": ["customers"]},
                )
            ),
            _final("I should query the customer table next."),
            _final("I still have not queried the database."),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("How many customers are there?")

    guard_text = "The task has not yet produced or successfully executed SQL."
    assert result.termination_reason is FunctionCallingTerminationReason.UNRESOLVED_COMPLETION
    assert result.protocol_completed is False
    assert result.completion_kind is CompletionKind.TERMINATED
    assert result.total_llm_turns == 3
    assert len(fake.requests) == 3
    assert sum(
        guard_text in (request.messages[0].content or "")
        for request in fake.requests
    ) == 1
    assert all(
        guard_text not in (message.content or "")
        for request in fake.requests
        for message in request.messages
        if message.role == "user"
    )
    assert result.steps[1].completion_guard_triggered is True
    assert result.steps[1].completion_guard_instruction is not None
    assert result.steps[-1].completion_guard_triggered is False
    assert all(
        guard_text not in summary
        for step in result.steps
        for summary in step.input_message_summary
    )


def test_final_sql_before_plain_answer_does_not_trigger_guard(demo_db: Path) -> None:
    sql = "SELECT COUNT(*) FROM customers"
    fake = FakeLLMClient(
        [
            _calls(_tool("validate", "validate_sql", {"sql": sql})),
            _final("The SQL is ready."),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("How many customers are there?")

    assert result.termination_reason is FunctionCallingTerminationReason.COMPLETED
    assert result.final_sql == sql
    assert result.execution_success is False
    assert len(fake.requests) == 2
    assert all(
        "The task has not yet produced" not in (message.content or "")
        for request in fake.requests
        for message in request.messages
    )


def test_exploration_is_followed_by_a_complete_final_query(demo_db: Path) -> None:
    exploratory_sql = "SELECT customer_id FROM customers LIMIT 2"
    final_sql = "SELECT COUNT(*) AS customer_count FROM customers"
    fake = FakeLLMClient(
        [
            _calls(_tool("explore", "execute_sql", {"sql": exploratory_sql})),
            _calls(_tool("final-query", "execute_sql", {"sql": final_sql})),
            _final("There are 10 customers."),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("How many customers are there?")

    prompt = fake.requests[0].messages[0].content or ""
    executed_sql = [
        call.arguments["sql"]
        for step in result.steps
        for call in step.tool_calls
        if call.name == "execute_sql"
    ]
    assert "Exploratory SQL is allowed" in prompt
    assert "the last execute_sql call must directly return exactly" in prompt
    assert executed_sql == [exploratory_sql, final_sql]
    assert result.final_sql == final_sql
    assert result.execution_result is not None
    assert result.execution_result.rows == [[10]]


def test_final_query_contract_requires_every_requested_column(demo_db: Path) -> None:
    final_sql = (
        "SELECT customer_name, city FROM customers "
        "ORDER BY customer_id LIMIT 2"
    )
    fake = FakeLLMClient(
        [
            _calls(_tool("final-query", "execute_sql", {"sql": final_sql})),
            _final("Returned both requested fields."),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run(
        "Give the names and cities of the first two customers."
    )

    prompt = fake.requests[0].messages[0].content or ""
    assert "complete columns" in prompt
    assert "do not omit required fields" in prompt
    assert result.execution_result is not None
    assert result.execution_result.columns == ["customer_name", "city"]
    assert result.final_sql == final_sql


def test_top_k_contract_requires_the_exact_limit_in_final_sql(demo_db: Path) -> None:
    exploratory_sql = (
        "SELECT customer_name FROM customers ORDER BY customer_id LIMIT 10"
    )
    final_sql = "SELECT customer_name FROM customers ORDER BY customer_id LIMIT 5"
    fake = FakeLLMClient(
        [
            _calls(_tool("explore", "execute_sql", {"sql": exploratory_sql})),
            _calls(_tool("final-query", "execute_sql", {"sql": final_sql})),
            _final("Returned the requested top five."),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("List the first five customers.")

    prompt = fake.requests[0].messages[0].content or ""
    assert "use the exact requested LIMIT in that final SQL" in prompt
    assert result.final_sql == final_sql
    assert result.execution_result is not None
    assert result.execution_result.returned_row_count == 5


def test_contract_forbids_truncating_only_in_natural_language(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [_final("No database answer."), _final("Still no database answer.")]
    )

    FunctionCallingAgent(demo_db, fake).run("List the first five customers.")

    prompt = fake.requests[0].messages[0].content or ""
    assert "do not return extra columns or extra rows" in prompt
    assert "never fetch more rows and truncate them only in natural language" in prompt


def test_second_native_http_request_replays_assistant_call_and_observation(
    demo_db: Path,
) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "schema",
                                        "type": "function",
                                        "function": {
                                            "name": "inspect_schema",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "完成"}, "finish_reason": "stop"}
                ]
            },
        )

    client = OpenAICompatibleLLMClient(
        OpenAICompatibleSettings(
            api_key="test", base_url="https://llm.example", model="test"
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = FunctionCallingAgent(demo_db, client).run("查看结构")

    assert result.protocol_completed is False
    assert result.termination_reason is FunctionCallingTerminationReason.UNRESOLVED_COMPLETION
    assert len(bodies) == 3
    second_messages = bodies[1]["messages"]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["tool_calls"][0]["id"] == "schema"
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["tool_call_id"] == "schema"


def test_sql_error_allows_reinspect_then_repair(demo_db: Path) -> None:
    repaired_sql = "SELECT customer_name FROM customers ORDER BY customer_id LIMIT 2"
    fake = FakeLLMClient(
        [
            _calls(_tool("schema-1", "inspect_schema", {})),
            _calls(
                _tool(
                    "bad-sql",
                    "execute_sql",
                    {"sql": "SELECT customer_name FROM customer LIMIT 2"},
                )
            ),
            _calls(_tool("schema-2", "inspect_schema", {})),
            _calls(_tool("fixed-sql", "execute_sql", {"sql": repaired_sql})),
            _final("前两名客户是张伟和王芳。"),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("列出前两名客户")

    assert result.termination_reason is FunctionCallingTerminationReason.COMPLETED
    assert result.total_llm_turns == 5
    assert result.steps[1].tool_results[0].error_code == "execution_error"
    assert result.steps[2].tool_results[0].success is True
    assert result.execution_result is not None
    assert result.execution_result.rows == [["张伟"], ["王芳"]]


def test_multiple_tool_calls_preserve_order_and_ids(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(
                _tool("first", "inspect_schema", {"table_names": ["customers"]}),
                _tool("second", "execute_sql", {"sql": "SELECT 1 AS value"}),
            ),
            _final(),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("测试多个工具")

    first_step = result.steps[0]
    assert [item.tool_call_id for item in first_step.tool_results] == ["first", "second"]
    assert [item.tool_call_id for item in first_step.tool_messages] == ["first", "second"]
    assert result.tool_call_count == 2


def test_same_turn_identical_call_is_rejected(demo_db: Path) -> None:
    repeated = {"table_names": ["customers"]}
    fake = FakeLLMClient(
        [
            _calls(
                _tool("first", "inspect_schema", repeated),
                _tool("second", "inspect_schema", repeated),
            )
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("重复调用")

    assert result.termination_reason is FunctionCallingTerminationReason.REPEATED_TOOL_CALL
    assert result.steps[0].tool_results[0].executed is True
    assert result.steps[0].tool_results[1].executed is False


def test_repeated_execute_sql_is_not_run_twice(demo_db: Path) -> None:
    sql = "SELECT COUNT(*) FROM customers"
    fake = FakeLLMClient(
        [
            _calls(_tool("one", "execute_sql", {"sql": sql})),
            _calls(_tool("two", "execute_sql", {"sql": " select COUNT(*) from CUSTOMERS ;"})),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("客户数量")

    assert result.termination_reason is FunctionCallingTerminationReason.REPEATED_SQL
    assert result.sql_execution_count == 1
    assert result.steps[1].tool_results[0].executed is False
    assert result.steps[1].tool_results[0].error_code == "repeated_sql"


def test_repeated_unknown_tool_terminates(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(_tool("one", "unknown_tool", {})),
            _calls(_tool("two", "unknown_tool", {})),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("测试未知工具")

    assert result.termination_reason is FunctionCallingTerminationReason.REPEATED_TOOL_CALL
    assert all(not step.tool_results[0].executed for step in result.steps)


def test_unknown_tool_and_invalid_parameters_can_be_corrected(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(_tool("unknown", "query_database", {})),
            _calls(_tool("invalid", "execute_sql", {})),
            _calls(_tool("fixed", "execute_sql", {"sql": "SELECT 1"})),
            _final(),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("纠正调用")

    assert result.termination_reason is FunctionCallingTerminationReason.COMPLETED
    assert result.steps[0].tool_results[0].error_code == "unknown_tool"
    assert result.steps[1].tool_results[0].error_code == "invalid_arguments"
    assert result.execution_success is True


def test_first_unsafe_sql_can_be_repaired(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(
                _tool("unsafe", "execute_sql", {"sql": "DELETE FROM customers"})
            ),
            _calls(_tool("safe", "execute_sql", {"sql": "SELECT COUNT(*) FROM customers"})),
            _final(),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("客户数量")

    assert result.termination_reason is FunctionCallingTerminationReason.COMPLETED
    assert result.steps[0].tool_results[0].executed is False
    assert result.steps[0].tool_results[0].error_code == "unsafe_sql"
    assert result.sql_execution_count == 1


def test_second_unsafe_candidate_terminates(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(
                _tool("unsafe-1", "validate_sql", {"sql": "DELETE FROM customers"})
            ),
            _calls(_tool("unsafe-2", "execute_sql", {"sql": "DROP TABLE customers"})),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("危险操作")

    assert result.termination_reason is FunctionCallingTerminationReason.UNSAFE_SQL
    assert result.sql_execution_count == 0
    assert all(not step.tool_results[0].executed for step in result.steps)


def test_repeated_inspect_without_new_error_terminates(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(_tool("one", "inspect_schema", {})),
            _calls(_tool("two", "inspect_schema", {})),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("查看结构")

    assert result.termination_reason is FunctionCallingTerminationReason.REPEATED_TOOL_CALL
    assert result.steps[1].tool_results[0].executed is False


def test_repeated_validate_without_new_observation_terminates(demo_db: Path) -> None:
    arguments = {"sql": "SELECT 1"}
    fake = FakeLLMClient(
        [
            _calls(_tool("one", "validate_sql", arguments)),
            _calls(_tool("two", "validate_sql", arguments)),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("重复验证")

    assert result.termination_reason is FunctionCallingTerminationReason.REPEATED_TOOL_CALL
    assert result.steps[1].tool_results[0].executed is False


def test_max_steps_terminates_without_final_answer(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(_tool("one", "inspect_schema", {"table_names": ["customers"]})),
            _calls(_tool("two", "inspect_schema", {"table_names": ["products"]})),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake, max_steps=2).run("持续查看结构")

    assert result.termination_reason is FunctionCallingTerminationReason.MAX_STEPS_REACHED
    assert result.protocol_completed is False
    assert result.completion_kind is CompletionKind.TERMINATED


def test_direct_answer_without_sql_is_guarded_then_unresolved(demo_db: Path) -> None:
    fake = FakeLLMClient([_final("直接回答"), _final("仍然直接回答")])

    result = FunctionCallingAgent(demo_db, fake).run("需要数据库回答")

    assert result.termination_reason is FunctionCallingTerminationReason.UNRESOLVED_COMPLETION
    assert result.protocol_completed is False
    assert result.execution_success is False
    assert result.answer_grounded is False
    assert result.result_correct is None
    assert result.completion_kind is CompletionKind.TERMINATED
    assert len(fake.requests) == 2


def test_json_fallback_responses_use_same_agent_loop(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            _calls(
                _tool("execute", "execute_sql", {"sql": "SELECT 1"}), fallback=True
            ),
            _final("完成", fallback=True),
        ]
    )

    result = FunctionCallingAgent(demo_db, fake).run("fallback")

    assert result.termination_reason is FunctionCallingTerminationReason.COMPLETED
    assert result.steps[0].llm_response is not None
    assert result.steps[0].llm_response.response_mode is ResponseMode.JSON_FALLBACK


def test_model_error_terminates(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [LLMError(code="provider_down", message="offline", retryable=True)]
    )

    result = FunctionCallingAgent(demo_db, fake).run("失败")

    assert result.termination_reason is FunctionCallingTerminationReason.MODEL_ERROR
    assert result.steps[0].llm_error is not None
