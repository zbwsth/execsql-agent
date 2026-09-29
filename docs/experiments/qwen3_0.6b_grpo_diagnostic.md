# Qwen3-0.6B non-thinking GRPO diagnostic

Date: 2026-09-18

This was a diagnostic smoke test of the verl/vLLM/RLVR plumbing. It was not a
formal model-quality or benchmark experiment.

## Reproduction

The run used the one-row DEV-only `smoke_sql_008` diagnostic dataset generated
by:

```bash
python -m training.build_grpo_smoke_dataset
```

The Qwen3 chat template used its official non-thinking mode. The DEV smoke
prompt also contained a strict JSON output contract. No completion
post-processing was used, and the reward function and SQL verifier were not
changed.

```bash
conda activate verl-v0.9.0-clean
export MODEL_PATH=/path/to/Qwen3-0.6B
export VLLM_USE_FLASHINFER_SAMPLER=0
bash scripts/run_verl_grpo_diagnostic.sh
```

The run is intentionally one optimizer step, uses four rollouts per prompt, and
does not save a checkpoint. Generation is stochastic, so an exact reward vector
is evidence from the recorded run rather than a promise for every rerun.

## Recorded result

The successful group produced:

```text
rewards:     [1.0, 0.2, 1.0, 0.2]
group mean:  0.6
sample std:  0.46188
advantages:  [+0.866, -0.866, +0.866, -0.866]
policy loss: -0.07949
grad norm:   17.49
global step: 0 -> 1
```

The finite non-zero gradient was followed by `optimizer.step()` and rollout
weight synchronization. This demonstrates a real parameter-update path, not a
formal effectiveness result.
