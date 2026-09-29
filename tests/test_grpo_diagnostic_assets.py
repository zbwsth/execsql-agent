"""Contract tests for the reproducible non-thinking GRPO diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from training.build_grpo_dataset import build_messages
from training.build_grpo_smoke_dataset import (
    DIAGNOSTIC_SAMPLE_ID,
    SMOKE_CASES,
    SMOKE_OUTPUT_CONTRACT,
    SmokeBuildResult,
    build_smoke_assets,
)
from training.verl_grpo_adapter import read_verl_parquet


@pytest.fixture(scope="module")
def diagnostic_assets(
    tmp_path_factory: pytest.TempPathFactory,
) -> SmokeBuildResult:
    return build_smoke_assets(tmp_path_factory.mktemp("grpo-diagnostic"))


def _prompt_text(record: Mapping[str, object]) -> str:
    prompt = record["prompt"]
    assert isinstance(prompt, list)
    contents: list[str] = []
    for message in prompt:
        assert isinstance(message, Mapping)
        contents.append(str(message["content"]))
    return "\n".join(contents)


def test_smoke_prompts_are_private_and_contain_output_contract(
    diagnostic_assets: SmokeBuildResult,
) -> None:
    records = read_verl_parquet(diagnostic_assets.train_path)
    records += read_verl_parquet(diagnostic_assets.val_path)

    assert len(records) == len(SMOKE_CASES)
    for record, case in zip(records, SMOKE_CASES, strict=True):
        prompt = _prompt_text(record)
        normalized = prompt.casefold()
        assert "gold_sql" not in normalized
        assert "expected_result" not in normalized
        assert case.reference_sql.casefold() not in normalized
        assert SMOKE_OUTPUT_CONTRACT in prompt


def test_output_contract_is_scoped_to_dev_smoke_builder() -> None:
    messages = build_messages(
        question="Test question",
        schema_context="CREATE TABLE example(id INTEGER);",
        domain_context="Formal training context.",
    )
    assert all(SMOKE_OUTPUT_CONTRACT not in message.content for message in messages)


def test_diagnostic_dataset_selects_observed_variance_case(
    diagnostic_assets: SmokeBuildResult,
) -> None:
    records = read_verl_parquet(diagnostic_assets.diagnostic_path)
    assert len(records) == 1
    extra_info = records[0]["extra_info"]
    assert isinstance(extra_info, Mapping)
    assert extra_info["sample_id"] == DIAGNOSTIC_SAMPLE_ID
    prompt = _prompt_text(records[0])
    assert SMOKE_OUTPUT_CONTRACT in prompt


def test_diagnostic_config_loads_non_thinking_settings() -> None:
    config = yaml.safe_load(Path("configs/verl_grpo_sql_diagnostic.yaml").read_text())
    script = Path("scripts/run_verl_grpo_diagnostic.sh").read_text()

    assert config["data"]["apply_chat_template_kwargs"]["enable_thinking"] is False
    assert config["data"]["train_files"] == "data/grpo_smoke/diagnostic.parquet"
    assert config["algorithm"]["adv_estimator"] == "grpo"
    assert config["actor"]["use_remove_padding"] is False
    assert config["actor"]["attn_implementation"] == "sdpa"
    assert config["actor"]["ppo_mini_batch_size"] == 1
    assert config["actor"]["ppo_micro_batch_size_per_gpu"] == 1
    assert config["actor"]["fsdp_param_offload"] is True
    assert config["actor"]["fsdp_optimizer_offload"] is True
    assert config["rollout"]["name"] == "vllm"
    assert config["rollout"]["n"] == 4
    assert config["rollout"]["temperature"] == 0.7
    assert config["rollout"]["top_p"] == 0.8
    assert config["rollout"]["top_k"] == 20
    assert config["rollout"]["gpu_memory_utilization"] == 0.35
    assert config["trainer"]["total_training_steps"] == 1
    assert config["trainer"]["save_freq"] == -1
    for required in (
        "VLLM_USE_FLASHINFER_SAMPLER=0",
        "MODEL_PATH",
        "+data.apply_chat_template_kwargs.enable_thinking=False",
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
    ):
        assert required in script
