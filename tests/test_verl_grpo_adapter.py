"""CPU-only contract tests for the verl v0.9.0 SQL GRPO boundary."""

from __future__ import annotations

import inspect
import json
import sqlite3
import statistics
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from training.build_grpo_dataset import load_samples
from training.build_grpo_smoke_dataset import (
    SMOKE_CASES,
    SmokeBuildResult,
    build_smoke_assets,
    build_smoke_samples,
)
from training.verl_grpo_adapter import (
    VERL_RECORD_KEYS,
    VERL_TARGET_VERSION,
    read_verl_parquet,
    sample_to_verl_record,
)
from training.verl_sql_reward import compute_score


@pytest.fixture(scope="module")
def smoke_assets(tmp_path_factory: pytest.TempPathFactory) -> SmokeBuildResult:
    return build_smoke_assets(tmp_path_factory.mktemp("verl-sql-smoke"))


def _first_record(smoke_assets: SmokeBuildResult) -> dict[str, object]:
    return read_verl_parquet(smoke_assets.train_path)[0]


def _reward(
    record: Mapping[str, object],
    completion: str,
    *,
    database_root: Path,
) -> dict[str, object]:
    reward_model = record["reward_model"]
    extra_info = record["extra_info"]
    assert isinstance(reward_model, Mapping)
    assert isinstance(extra_info, Mapping)
    return compute_score(
        str(record["data_source"]),
        completion,
        reward_model["ground_truth"],
        extra_info,
        database_root=str(database_root),
    )


def test_grpo_sample_maps_to_verl_v090_record(smoke_assets: SmokeBuildResult) -> None:
    samples = build_smoke_samples(
        smoke_assets.database_path,
        metadata_database_path=str(smoke_assets.database_path),
    )
    record = sample_to_verl_record(samples[0], index=0, split="train")

    assert VERL_TARGET_VERSION == "v0.9.0"
    assert set(record) == VERL_RECORD_KEYS
    assert record["ability"] == "sql"
    assert record["data_source"] == "execsql_sql_dev_smoke_v1"
    prompt = record["prompt"]
    assert isinstance(prompt, list)
    roles: list[object] = []
    for message in prompt:
        assert isinstance(message, Mapping)
        roles.append(message["role"])
    assert roles == ["system", "user"]
    reward_model = record["reward_model"]
    assert isinstance(reward_model, Mapping)
    private = json.loads(str(reward_model["ground_truth"]))
    assert private["verifier_type"] == "sqlite_execution_result_v1"
    assert "expected_result" in private


def test_parquet_has_no_gold_sql_and_prompt_has_no_expected_result(
    smoke_assets: SmokeBuildResult,
) -> None:
    records = read_verl_parquet(smoke_assets.train_path) + read_verl_parquet(smoke_assets.val_path)
    assert len(records) == len(SMOKE_CASES) == 12
    for record in records:
        serialized = json.dumps(record, ensure_ascii=False, default=str).casefold()
        prompt = json.dumps(record["prompt"], ensure_ascii=False).casefold()
        assert "gold_sql" not in serialized
        assert "expected_result" not in prompt
        assert "database_path" not in prompt


def test_reward_callback_signature_matches_verl_v090() -> None:
    parameters = list(inspect.signature(compute_score).parameters)
    assert parameters[:4] == [
        "data_source",
        "solution_str",
        "ground_truth",
        "extra_info",
    ]


def test_reward_callback_correct_sql_receives_one(
    smoke_assets: SmokeBuildResult,
) -> None:
    result = _reward(
        _first_record(smoke_assets),
        '{"sql":"SELECT COUNT(*) AS customer_count FROM customers"}',
        database_root=smoke_assets.database_path.parent,
    )
    assert result["score"] == 1.0
    assert result["comparison_status"] == "matched"


def test_reward_callback_executable_wrong_sql_receives_point_two(
    smoke_assets: SmokeBuildResult,
) -> None:
    result = _reward(
        _first_record(smoke_assets),
        '{"sql":"SELECT COUNT(*) AS customer_count FROM products"}',
        database_root=smoke_assets.database_path.parent,
    )
    assert result["score"] == 0.2
    assert result["comparison_status"] == "mismatched"


@pytest.mark.parametrize(
    ("completion", "failure_kind"),
    [
        ("not SQL or JSON", "malformed_response"),
        ('{"sql":"DELETE FROM customers"}', "unsafe_sql"),
    ],
)
def test_reward_callback_malformed_and_unsafe_receive_zero(
    smoke_assets: SmokeBuildResult,
    completion: str,
    failure_kind: str,
) -> None:
    result = _reward(
        _first_record(smoke_assets),
        completion,
        database_root=smoke_assets.database_path.parent,
    )
    assert result["score"] == 0.0
    assert result["failure_kind"] == failure_kind


def test_completion_cannot_inject_database_path(smoke_assets: SmokeBuildResult) -> None:
    result = _reward(
        _first_record(smoke_assets),
        '{"sql":"SELECT 1","database_path":"/tmp/attacker.db"}',
        database_root=smoke_assets.database_path.parent,
    )
    assert result["score"] == 0.0
    assert result["failure_kind"] == "response_database_path"
    with sqlite3.connect(smoke_assets.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM customers").fetchone() == (10,)


def test_smoke_dataset_exports_all_rows_and_private_metadata(
    smoke_assets: SmokeBuildResult,
) -> None:
    train = read_verl_parquet(smoke_assets.train_path)
    val = read_verl_parquet(smoke_assets.val_path)
    assert smoke_assets.train_count == len(train) == 10
    assert smoke_assets.val_count == len(val) == 2
    for expected_split, records in (("train", train), ("val", val)):
        for record in records:
            extra_info = record["extra_info"]
            reward_model = record["reward_model"]
            assert isinstance(extra_info, Mapping)
            assert isinstance(reward_model, Mapping)
            assert extra_info["split"] == expected_split
            assert reward_model["ground_truth"]


def test_mock_four_response_group_has_nonzero_reward_variance(
    smoke_assets: SmokeBuildResult,
) -> None:
    record = _first_record(smoke_assets)
    completions = [
        '{"sql":"SELECT COUNT(*) AS customer_count FROM customers"}',
        '{"sql":"SELECT COUNT(*) AS customer_count FROM products"}',
        "not SQL",
        '{"sql":"DROP TABLE customers"}',
    ]
    rewards: list[float] = []
    for completion in completions:
        score = _reward(
            record,
            completion,
            database_root=smoke_assets.database_path.parent,
        )["score"]
        assert isinstance(score, int | float)
        rewards.append(float(score))
    assert rewards == [1.0, 0.2, 0.0, 0.0]
    assert statistics.pstdev(rewards) > 0


def test_smoke_config_and_command_pin_required_v090_fields() -> None:
    config = yaml.safe_load(Path("configs/verl_grpo_sql_smoke.yaml").read_text())
    script = Path("scripts/run_verl_grpo_smoke.sh").read_text()

    assert config["target"]["verl_tag"] == "v0.9.0"
    assert config["target"]["model"] == "Qwen/Qwen3-0.6B"
    assert config["algorithm"]["adv_estimator"] == "grpo"
    assert config["rollout"]["name"] == "vllm"
    assert config["rollout"]["n"] == 4
    assert config["trainer"]["total_training_steps"] == 2
    for required in (
        "python3 -m verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=4",
        "reward.custom_reward_function.path=",
        "reward.custom_reward_function.name=compute_score",
        "trainer.total_training_steps=2",
        "trainer.logger='[\"console\"]'",
    ):
        assert required in script
    assert "Qwen/Qwen3-8B" not in script


def test_generic_export_requires_an_existing_database(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError, match="Database does not exist"):
        load_samples(
            dataset_path=Path("data/synthetic/eval_questions.json"),
            database_path=missing,
        )
