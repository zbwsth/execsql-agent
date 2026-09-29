"""CPU-only contracts for the BIRD VERL v0.9.0 agentic integration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import (
    ExpectedResult,
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
    ToolCallResult,
)
from execsql_agent.rlvr.models import (
    DatabaseMetadata,
    ExtraInfo,
    GRPOSample,
    PromptMessage,
    VerifierMetadata,
)
from execsql_agent.rlvr.verifier import SQLExecutionVerifier
from execsql_agent.tools.registry import ToolRegistry
from training.verl_bird_agentic_adapter import (
    BIRD_GRPO_DATA_SOURCE,
    HARNESS_V2_MAX_STEPS,
    HARNESS_V2_SYSTEM_PROMPT,
    HARNESS_V2_TOOL_NAMES,
    sample_to_agentic_record,
    write_agentic_parquet,
)
from training.verl_bird_reward import compute_score
from training.verl_grpo_adapter import read_verl_parquet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(
    database: Path,
    *,
    archive_path: str = "data/bird/train/raw/train_databases.zip",
) -> GRPOSample:
    db_id = database.stem
    question = "How many customers are there?"
    return GRPOSample(
        sample_id="bird_train_contract_000",
        data_source=BIRD_GRPO_DATA_SOURCE,
        question=question,
        messages=[
            PromptMessage(role="system", content=HARNESS_V2_SYSTEM_PROMPT),
            PromptMessage(role="user", content=question),
        ],
        database_metadata=DatabaseMetadata(
            database_id=db_id,
            database_path=(f"data/bird/train/runtime_databases/{db_id}/{db_id}.sqlite"),
            database_sha256=_sha256(database),
            database_archive_path=archive_path,
            database_archive_member=(f"train_databases/{db_id}/{db_id}.sqlite"),
        ),
        private_verifier_metadata=VerifierMetadata(
            response_contract="agentic_tool_loop_v1",
            expected_result=ExpectedResult(
                columns=["count"],
                rows=[[10]],
                ordered=False,
            ),
        ),
        extra_info=ExtraInfo(
            dataset_name=BIRD_GRPO_DATA_SOURCE,
            evaluation_group="contract_test",
            prompt_version="grpo_agentic_harness_v2",
        ),
    )


def _agentic_transcript(sql: str) -> str:
    return (
        '<tool_call>{"name":"list_tables","arguments":{}}</tool_call>'
        '{"tables":["customers"]}'
        '<tool_call>{"name":"inspect_schema","arguments":'
        '{"table_names":["customers"]}}</tool_call>'
        '{"database_id":"demo"}'
        '<tool_call>{"name":"execute_sql","arguments":{"sql":' + json.dumps(sql) + "}}</tool_call>"
        '{"execution_success":true}'
        "There are ten customers."
    )


def test_policy_messages_match_frozen_harness_prompt(demo_db: Path) -> None:
    fake = FakeLLMClient(
        [
            LLMResponse(
                final_answer="No database answer yet.",
                response_mode=ResponseMode.PLAIN_FINAL,
            ),
            LLMResponse(
                final_answer="Unable to complete.",
                response_mode=ResponseMode.PLAIN_FINAL,
            ),
        ]
    )
    question = "How many customers are there?"
    FunctionCallingAgent(demo_db, fake, max_steps=HARNESS_V2_MAX_STEPS).run(question)

    first = fake.requests[0]
    assert [message.role for message in first.messages] == ["system", "user"]
    assert first.messages[0].content == HARNESS_V2_SYSTEM_PROMPT
    assert first.messages[1].content == question
    assert {tool.name for tool in first.tools} == set(HARNESS_V2_TOOL_NAMES)


def test_agentic_record_enforces_private_policy_boundary(demo_db: Path) -> None:
    sample = _sample(demo_db)
    record = sample_to_agentic_record(
        sample,
        index=0,
        split="train",
        gold_sql_for_scan="SELECT COUNT(*) AS count FROM customers",
        evidence_for_scan="private BIRD evidence",
    )

    prompt = json.dumps(record["prompt"], ensure_ascii=False).casefold()
    serialized = json.dumps(record, ensure_ascii=False).casefold()
    assert "expected_result" not in prompt
    assert "database_path" not in prompt
    assert "private bird evidence" not in serialized
    assert "gold_sql" not in serialized
    extra_info = record["extra_info"]
    assert isinstance(extra_info, Mapping)
    assert extra_info["agent_name"] == "tool_agent"
    assert extra_info["need_tools_kwargs"] is True
    assert extra_info["tool_selection"] == list(HARNESS_V2_TOOL_NAMES)


def test_agentic_parquet_round_trip_preserves_private_boundary(
    demo_db: Path,
    tmp_path: Path,
) -> None:
    record = sample_to_agentic_record(
        _sample(demo_db),
        index=0,
        split="train",
        gold_sql_for_scan="SELECT COUNT(*) AS count FROM customers",
        evidence_for_scan="private evidence",
    )
    output = write_agentic_parquet([record], tmp_path / "agentic.parquet")
    rows = read_verl_parquet(output)

    assert len(rows) == 1
    row = rows[0]
    prompt = json.dumps(row["prompt"], ensure_ascii=False).casefold()
    assert "expected_result" not in prompt
    ground_truth = json.loads(row["reward_model"]["ground_truth"])
    assert ground_truth["response_contract"] == "agentic_tool_loop_v1"
    assert ground_truth["expected_result"]["rows"] == [[10]]
    tools_kwargs = row["extra_info"]["tools_kwargs"]
    assert set(tools_kwargs) == set(HARNESS_V2_TOOL_NAMES)
    assert {value["create_kwargs"]["database_id"] for value in tools_kwargs.values()} == {
        demo_db.stem
    }


def test_agentic_verifier_uses_last_execute_sql(demo_db: Path) -> None:
    expected = ExpectedResult(columns=["count"], rows=[[10]], ordered=False)
    transcript = _agentic_transcript(
        "SELECT COUNT(*) AS count FROM products"
    ) + _agentic_transcript("SELECT COUNT(*) AS count FROM customers")
    verified = SQLExecutionVerifier(demo_db).verify(
        transcript,
        expected,
        response_contract="agentic_tool_loop_v1",
    )

    assert verified.reward == 1.0
    assert verified.sql == "SELECT COUNT(*) AS count FROM customers"


def test_r1_r2_views_and_archive_routing_are_deterministic(
    demo_db: Path,
    tmp_path: Path,
) -> None:
    db_id = demo_db.stem
    archive = tmp_path / "train_databases.zip"
    member = f"train_databases/{db_id}/{db_id}.sqlite"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as stream:
        stream.write(demo_db, member)
    sample = _sample(demo_db, archive_path=archive.name)
    record = sample_to_agentic_record(sample, index=0, split="train")
    reward_model = record["reward_model"]
    extra_info = record["extra_info"]
    assert isinstance(reward_model, Mapping)
    assert isinstance(extra_info, Mapping)

    correct = _agentic_transcript("SELECT COUNT(*) AS count FROM customers")
    wrong = _agentic_transcript("SELECT COUNT(*) AS count FROM products")
    r1_correct = compute_score(
        BIRD_GRPO_DATA_SOURCE,
        correct,
        reward_model["ground_truth"],
        extra_info,
        database_root=str(tmp_path),
        temporary_root=str(tmp_path),
        reward_view="r1_exact",
    )
    r1_wrong = compute_score(
        BIRD_GRPO_DATA_SOURCE,
        wrong,
        reward_model["ground_truth"],
        extra_info,
        database_root=str(tmp_path),
        temporary_root=str(tmp_path),
        reward_view="r1_exact",
    )
    r2_wrong = compute_score(
        BIRD_GRPO_DATA_SOURCE,
        wrong,
        reward_model["ground_truth"],
        extra_info,
        database_root=str(tmp_path),
        temporary_root=str(tmp_path),
        reward_view="r2_execution_shaped",
    )
    repeated = compute_score(
        BIRD_GRPO_DATA_SOURCE,
        correct,
        reward_model["ground_truth"],
        extra_info,
        database_root=str(tmp_path),
        temporary_root=str(tmp_path),
        reward_view="r1_exact",
    )

    assert r1_correct["score"] == r1_correct["r1_score"] == 1.0
    assert r1_wrong["score"] == r1_wrong["r1_score"] == 0.0
    assert r1_wrong["r2_score"] == r2_wrong["score"] == 0.2
    assert repeated == r1_correct


def test_e2_contract_pins_native_tool_agent_and_six_steps() -> None:
    config = yaml.safe_load(Path("configs/verl_grpo_bird_agentic_diagnostic.yaml").read_text())
    assert "ppo_trainer" in config["defaults"]
    actor_rollout_ref = config["actor_rollout_ref"]
    model = actor_rollout_ref["model"]
    rollout = actor_rollout_ref["rollout"]
    assert model["lora_rank"] == 16
    assert model["lora_alpha"] == 32
    assert model["path"] == "${oc.env:MODEL_PATH,/path/to/Qwen3-8B}"
    assert model["lora_adapter_path"] == (
        "${oc.env:SFT_ADAPTER_PATH,/path/to/sft_adapter}"
    )
    assert rollout["agent"]["default_agent_loop"] == "tool_agent"
    assert rollout["multi_turn"]["enable"] is True
    assert rollout["multi_turn"]["max_assistant_turns"] == 6
    assert rollout["multi_turn"]["max_user_turns"] == 6
    assert rollout["multi_turn"]["format"] == "hermes"
    assert rollout["val_kwargs"]["n"] == 4
    assert rollout["calculate_log_probs"] is False
    assert config["trainer"]["val_only"] is True
    assert config["trainer"]["val_before_train"] is True
    reward_kwargs = config["reward"]["custom_reward_function"]["reward_kwargs"]
    assert reward_kwargs["reward_view"] == "r2_execution_shaped"
    assert config["reward"]["reward_model"]["enable"] is False


def test_verl_v090_recognizes_sft_adapter_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verl = pytest.importorskip("verl")
    from verl.workers.config.model import HFModelConfig

    verl_root = Path(verl.__file__).resolve().parent
    adapter = tmp_path / "sft_adapter"
    adapter.mkdir()
    adapter_config_path = adapter / "adapter_config.json"
    adapter_config_path.write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(HFModelConfig, "__post_init__", lambda self: None)
    model_config = HFModelConfig(
        path=str(tmp_path / "Qwen3-8B"),
        load_tokenizer=False,
        lora_rank=adapter_config["r"],
        lora_alpha=adapter_config["lora_alpha"],
        target_modules=adapter_config["target_modules"],
        lora_adapter_path=str(adapter),
    )
    source = (verl_root / "workers/engine/fsdp/transformer_impl.py").read_text(
        encoding="utf-8"
    )

    assert verl.__version__ == "0.9.0"
    assert model_config.lora_adapter_path == str(adapter)
    assert model_config.lora_rank == 16
    assert model_config.lora_alpha == 32
    assert "PeftModel.from_pretrained" in source
    assert "is_trainable=True" in source


def test_verl_tool_wrapper_matches_registry_and_routes_database(
    demo_db: Path,
    tmp_path: Path,
) -> None:
    pytest.importorskip("verl")
    from training.verl_bird_tools import BirdToolRegistryTool

    db_id = demo_db.stem
    archive = tmp_path / "train_databases.zip"
    member = f"train_databases/{db_id}/{db_id}.sqlite"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as stream:
        stream.write(demo_db, member)
    tool = BirdToolRegistryTool(
        config={
            "type": "native",
            "name": "list_tables",
            "database_archive": str(archive),
            "temporary_root": str(tmp_path),
        },
        tool_schema=None,
    )
    create_kwargs = {
        "database_id": db_id,
        "database_path": (f"data/bird/train/runtime_databases/{db_id}/{db_id}.sqlite"),
        "database_sha256": _sha256(demo_db),
        "database_archive_member": member,
    }

    async def exercise() -> ToolCallResult:
        instance_id, _ = await tool.create(create_kwargs=create_kwargs)
        try:
            response, reward, metrics = await tool.execute(
                instance_id,
                {},
            )
            assert reward == 0.0
            assert metrics["database_id"] == db_id
            assert response.text is not None
            return ToolCallResult.model_validate_json(response.text)
        finally:
            await tool.release(instance_id)

    wrapped = asyncio.run(exercise())
    _, direct = ToolRegistry(demo_db).dispatch(
        ToolCallRequest(id="direct", name="list_tables", arguments={})
    )
    assert wrapped.success == direct.success
    assert wrapped.executed == direct.executed
    assert wrapped.output == direct.output
    schema = tool.tool_schema.model_dump(mode="json")
    expected = next(
        item for item in ToolRegistry(demo_db).definitions if item.name == "list_tables"
    )
    assert schema == {
        "type": "function",
        "function": expected.model_dump(mode="json"),
    }


def test_all_verl_tool_schemas_exactly_match_harness() -> None:
    pytest.importorskip("verl")
    from training.verl_bird_tools import BirdToolRegistryTool

    registry = ToolRegistry(Path("/schema-only.sqlite"))
    definitions = {item.name: item for item in registry.definitions}
    for name in HARNESS_V2_TOOL_NAMES:
        tool = BirdToolRegistryTool(
            config={
                "type": "native",
                "name": name,
                "database_archive": "/not-opened.zip",
                "temporary_root": "/not-created",
            },
            tool_schema=None,
        )
        assert tool.tool_schema.model_dump(mode="json") == {
            "type": "function",
            "function": definitions[name].model_dump(mode="json"),
        }


def test_verl_native_tool_observation_tokens_are_zero_masked() -> None:
    pytest.importorskip("verl")
    from verl.experimental.agent_loop.tool_agent_loop import (
        AgentData,
        AgentState,
        ToolAgentLoop,
    )
    from verl.experimental.agent_loop.tool_parser import FunctionCall
    from verl.tools.function_tool import FunctionTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

    definition = next(
        item
        for item in ToolRegistry(Path("/schema-only.sqlite")).definitions
        if item.name == "list_tables"
    )
    schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": definition.model_dump(mode="json"),
        }
    )

    async def call() -> ToolResponse:
        return ToolResponse(text='{"tables":["customers"]}')

    function_tool = FunctionTool(
        name="list_tables",
        fn=call,
        tool_schema=schema,
        is_async=True,
    )
    loop = object.__new__(ToolAgentLoop)
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 4096
    loop.tool_response_truncate_side = "middle"
    loop.tools = {"list_tables": function_tool}
    loop.tool_parser_name = "hermes"
    loop.enable_continuous_token = False
    loop.turn_separator = []
    loop.processor = None
    loop.response_length = 4096

    async def apply_chat_template(*args: object, **kwargs: object) -> list[int]:
        del args, kwargs
        return [101, 102, 103]

    loop.apply_chat_template = apply_chat_template
    data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        audio_data=None,
        mm_processor_kwargs=None,
        metrics={},
        request_id="mask-test",
        tools_kwargs={},
    )
    data.prompt_ids = [1, 2]
    data.response_mask = [1, 1]
    data.tool_calls = [
        FunctionCall(
            name="list_tables",
            arguments="{}",
            tool_call_id="mask-call",
        )
    ]

    state = asyncio.run(loop._handle_processing_tools_state(data))

    assert state is AgentState.GENERATING
    assert data.response_mask == [1, 1, 0, 0, 0]
