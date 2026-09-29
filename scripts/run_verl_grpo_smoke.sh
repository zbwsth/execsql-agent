#!/usr/bin/env bash
# DEV / SMOKE ONLY: two-step verl v0.9.0 SQL GRPO plumbing check.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_FILE="${PROJECT_ROOT}/data/grpo_smoke/train.parquet"
VAL_FILE="${PROJECT_ROOT}/data/grpo_smoke/val.parquet"
DATABASE_FILE="${PROJECT_ROOT}/data/grpo_smoke/sql_smoke.db"
REWARD_FILE="${PROJECT_ROOT}/training/verl_sql_reward.py"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/outputs/verl_grpo_sql_smoke}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

for required_file in "${TRAIN_FILE}" "${VAL_FILE}" "${DATABASE_FILE}" "${REWARD_FILE}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing smoke prerequisite: ${required_file}" >&2
        echo "Run: python -m training.build_grpo_smoke_dataset" >&2
        exit 1
    fi
done

exec python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1280 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_model_len=1280 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1280 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1280 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1280 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.num_workers=1 \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="${REWARD_FILE}" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.database_root="${PROJECT_ROOT}" \
    trainer.use_v1=True \
    trainer.logger='["console"]' \
    trainer.project_name=execsql_agent \
    trainer.experiment_name=sql_grpo_dev_smoke_v1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.test_freq=1 \
    trainer.save_freq=1 \
    trainer.total_training_steps=2 \
    trainer.resume_mode=disable \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    "$@"
