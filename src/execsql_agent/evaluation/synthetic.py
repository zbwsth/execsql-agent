"""Load deterministic Synthetic datasets and build scripted FakeLLM clients."""

from __future__ import annotations

import json
from pathlib import Path

from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import (
    BehaviorDataset,
    EvaluationCase,
    EvaluationDataset,
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
)


def load_evaluation_dataset(path: str | Path) -> EvaluationDataset:
    """Load and strictly validate a normal Synthetic evaluation dataset."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvaluationDataset.model_validate(raw)


def load_behavior_dataset(path: str | Path) -> BehaviorDataset:
    """Load and strictly validate deterministic behavior scenarios."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return BehaviorDataset.model_validate(raw)


def build_scripted_client(case: EvaluationCase, agent_mode: str) -> FakeLLMClient:
    """Build a queue from explicit FakeLLM responses, never from expected results."""

    scripts = case.fake_responses
    if scripts is not None and agent_mode in scripts:
        return FakeLLMClient(scripts[agent_mode])
    if case.fake_sql is None:
        raise ValueError(
            f"Case {case.id!r} has no deterministic FakeLLM script for {agent_mode!r}."
        )
    if agent_mode == "pipeline":
        generation = json.dumps(
            {
                "sql": case.fake_sql,
                "reason": "Synthetic deterministic SQL candidate.",
                "referenced_tables": [],
                "referenced_columns": [],
            },
            ensure_ascii=False,
        )
        return FakeLLMClient(
            [
                LLMResponse(
                    final_answer=generation,
                    response_mode=ResponseMode.PLAIN_FINAL,
                )
            ]
        )
    return FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id=f"{case.id}_schema",
                        name="inspect_schema",
                        arguments={},
                    )
                ],
                response_mode=ResponseMode.NATIVE_TOOL_CALLS,
            ),
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id=f"{case.id}_execute",
                        name="execute_sql",
                        arguments={"sql": case.fake_sql},
                    )
                ],
                response_mode=ResponseMode.NATIVE_TOOL_CALLS,
            ),
            LLMResponse(
                final_answer="已根据真实 SQLite 工具结果完成回答。",
                response_mode=ResponseMode.PLAIN_FINAL,
            ),
        ]
    )
