"""Minimal Qwen3-8B QLoRA SFT pipeline with verified assistant-only labels."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import bitsandbytes as bnb
import torch
from assistant_turn_preprocessing import (
    ProcessedTurn,
    load_jsonl,
    process_trajectory,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.training_args import OptimizerNames

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_LENGTH = 4096


class SingleTurnDataset(Dataset[dict[str, list[int]]]):
    """A one-item torch dataset containing already-tokenized assistant-only labels."""

    def __init__(self, turn: ProcessedTurn) -> None:
        self._feature = {
            "input_ids": turn.input_ids,
            "attention_mask": turn.attention_mask,
            "labels": turn.labels,
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        if index != 0:
            raise IndexError(index)
        return self._feature


@dataclass(frozen=True)
class AssistantOnlyCollator:
    """Dynamically pad pretokenized inputs without regenerating their labels."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer requires pad_token_id")
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for feature in features:
            length = len(feature["input_ids"])
            if length != len(feature["attention_mask"]) or length != len(feature["labels"]):
                raise ValueError("Pretokenized feature lengths differ")
            padding = max_length - length
            batch_input_ids.append(feature["input_ids"] + [pad_token_id] * padding)
            batch_attention_mask.append(feature["attention_mask"] + [0] * padding)
            batch_labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def load_longest_train_turn(
    *,
    tokenizer: PreTrainedTokenizerBase,
    train_path: Path,
) -> tuple[ProcessedTurn, int, int]:
    """Preprocess all trajectories and return the longest assistant turn."""

    trajectories = load_jsonl(train_path)
    if not trajectories:
        raise ValueError("Training dataset has no trajectories")
    turns: list[ProcessedTurn] = []
    for sample in trajectories:
        turns.extend(
            process_trajectory(
                tokenizer=tokenizer,
                sample=sample,
                split="train",
                max_length=MAX_LENGTH,
            )
        )
    if not turns:
        raise ValueError("Training dataset has no assistant turns")
    longest = max(turns, key=lambda turn: turn.sequence_tokens)
    if longest.sequence_tokens > MAX_LENGTH:
        raise ValueError(f"Longest sequence exceeds {MAX_LENGTH}")
    if longest.supervised_tokens <= 0:
        raise ValueError("Smoke sample has no supervised assistant tokens")
    if not any(label != -100 for label in longest.labels):
        raise ValueError("Smoke sample labels contain no supervised token")
    if not any(label == -100 for label in longest.labels):
        raise ValueError("Smoke sample labels contain no masked prompt token")
    if longest.context_tokens_supervised != 0:
        raise ValueError("Smoke sample supervises prompt/context tokens")
    return longest, len(trajectories), len(turns)


def quantization_config() -> BitsAndBytesConfig:
    """Return the fixed NF4/BF16 double-quantization configuration."""

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_quantized_base(model_path: Path) -> torch.nn.Module:
    """Load the local base model on GPU 0 without automatic device placement."""

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config(),
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )


def inspect_quantization(model: torch.nn.Module) -> dict[str, object]:
    """Verify every bitsandbytes linear uses NF4 and BF16 computation."""

    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear4bit)
    ]
    quant_types = {str(getattr(module.weight, "quant_type", None)).lower() for _, module in modules}
    compute_dtypes = {str(module.compute_dtype) for _, module in modules}
    trainable_base_weights = [
        f"{name}.weight" for name, module in modules if module.weight.requires_grad
    ]
    result = {
        "is_loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "linear4bit_module_count": len(modules),
        "quant_types": sorted(quant_types),
        "compute_dtypes": sorted(compute_dtypes),
        "module_name_examples": [name for name, _ in modules[:12]],
        "trainable_base_4bit_weights": trainable_base_weights,
    }
    if result["is_loaded_in_4bit"] is not True:
        raise ValueError("Model is not marked as loaded in 4-bit")
    if not modules:
        raise ValueError("No bitsandbytes Linear4bit modules found")
    if quant_types != {"nf4"}:
        raise ValueError(f"Expected only NF4 weights, found {sorted(quant_types)}")
    if compute_dtypes != {"torch.bfloat16"}:
        raise ValueError(f"Expected only BF16 compute, found {sorted(compute_dtypes)}")
    if trainable_base_weights:
        raise ValueError("Some base 4-bit weights unexpectedly require gradients")
    return result


def attach_lora(model: torch.nn.Module) -> tuple[torch.nn.Module, dict[str, object]]:
    """Prepare the k-bit base and attach all-linear LoRA adapters."""

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    target_modules = [
        (name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)
    ]
    target_names = [name for name, _ in target_modules]
    target_types = sorted({type(module.get_base_layer()).__name__ for _, module in target_modules})
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected_trainable = [name for name in trainable_names if "lora_" not in name]
    lm_head = model.get_output_embeddings()
    lm_head_lora = isinstance(lm_head, LoraLayer) or hasattr(lm_head, "lora_A")
    if not target_modules:
        raise ValueError("LoRA did not attach to any module")
    if unexpected_trainable:
        raise ValueError(f"Non-LoRA parameters are trainable: {unexpected_trainable[:10]}")
    if lm_head_lora:
        raise ValueError("lm_head was unexpectedly LoRA-adapted")
    trainable_params, total_params = model.get_nb_trainable_parameters()
    if trainable_params <= 0 or total_params <= 0:
        raise ValueError("Invalid PEFT parameter counts")
    details = {
        "target_module_count": len(target_modules),
        "target_module_types": target_types,
        "target_module_name_examples": target_names[:20],
        "trainable_parameter_name_examples": trainable_names[:12],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": 100.0 * trainable_params / total_params,
        "unexpected_trainable_parameters": unexpected_trainable,
        "lm_head_is_lora": lm_head_lora,
    }
    return model, details


def cuda_memory() -> dict[str, int | str]:
    """Return current single-GPU memory counters in bytes."""

    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "total_bytes": properties.total_memory,
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
    }


def optimizer_supported() -> None:
    """Hard-stop unless this Transformers build exposes paged AdamW 8-bit."""

    available = {optimizer.value for optimizer in OptimizerNames}
    if "paged_adamw_8bit" not in available:
        raise ValueError(
            "Transformers OptimizerNames does not support paged_adamw_8bit; "
            f"available={sorted(available)}"
        )


def snapshot_lora_b(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy the first three deterministic LoRA-B parameters before the update."""

    selected = sorted(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_B" in name and parameter.requires_grad
    )[:3]
    if len(selected) != 3:
        raise ValueError(
            f"Expected at least three trainable lora_B parameters, found {len(selected)}"
        )
    return {name: parameter.detach().float().cpu().clone() for name, parameter in selected}


def lora_deltas(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float]:
    """Calculate max absolute update for each saved LoRA-B parameter."""

    current = dict(model.named_parameters())
    deltas: dict[str, float] = {}
    for name, old_value in before.items():
        parameter = current.get(name)
        if parameter is None:
            raise ValueError(f"LoRA parameter disappeared: {name}")
        delta = (parameter.detach().float().cpu() - old_value).abs().max().item()
        deltas[name] = float(delta)
    if not any(delta > 0 for delta in deltas.values()):
        raise ValueError("No checked LoRA-B parameter changed after optimizer step")
    return deltas


def find_step_log(log_history: list[dict[str, Any]]) -> dict[str, object]:
    """Return the first Trainer log containing the actual training loss."""

    for entry in log_history:
        if "loss" in entry:
            return dict(entry)
    raise ValueError("Trainer log history contains no training loss")


def save_adapter(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    output_path: Path,
) -> dict[str, object]:
    """Save only PEFT adapter and tokenizer after successful smoke training."""

    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty adapter path: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    files = sorted(path for path in output_path.rglob("*") if path.is_file())
    if any(path.name.startswith("model-") for path in files):
        raise ValueError("A full base-model shard was unexpectedly saved")
    return {
        "path": str(output_path),
        "files": [str(path.relative_to(output_path)) for path in files],
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def reload_adapter_forward(
    *,
    model_path: Path,
    adapter_path: Path,
    turn: ProcessedTurn,
) -> float:
    """Reload 4-bit base plus frozen adapter and run one finite-loss forward."""

    base_model = load_quantized_base(model_path)
    inspect_quantization(base_model)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Reloaded inference adapter unexpectedly has trainable parameters")
    model.eval()
    device = next(model.parameters()).device
    batch = {
        "input_ids": torch.tensor([turn.input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([turn.attention_mask], dtype=torch.long, device=device),
        "labels": torch.tensor([turn.labels], dtype=torch.long, device=device),
    }
    with torch.no_grad():
        outputs = model(**batch)
    loss = float(outputs.loss.detach().float().cpu().item())
    if not math.isfinite(loss):
        raise ValueError(f"Reload forward loss is not finite: {loss}")
    del outputs, batch, model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    return loss


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Qwen3-8B QLoRA optimizer step.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_steps != 1:
        raise ValueError("This smoke entry point requires --max-steps 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")
    optimizer_supported()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    smoke_turn, trajectory_count, turn_count = load_longest_train_turn(
        tokenizer=tokenizer,
        train_path=args.train,
    )
    smoke_summary = {
        "case_id": smoke_turn.case_id,
        "assistant_turn_index": smoke_turn.turn_index,
        "target_type": smoke_turn.target_type,
        "sequence_length": smoke_turn.sequence_tokens,
        "supervised_tokens": smoke_turn.supervised_tokens,
        "masked_tokens": smoke_turn.prompt_masked_tokens,
        "train_trajectories": trajectory_count,
        "assistant_turn_samples": turn_count,
    }
    print(json.dumps({"stage": "data_ready", "smoke_sample": smoke_summary}, ensure_ascii=False))

    torch.cuda.empty_cache()
    load_started = perf_counter()
    model = load_quantized_base(args.model)
    model_load_seconds = perf_counter() - load_started
    memory_after_4bit_load = cuda_memory()
    quantization = inspect_quantization(model)
    model, lora = attach_lora(model)
    lora_before = snapshot_lora_b(model)
    print(
        json.dumps(
            {
                "stage": "model_ready",
                "model_load_seconds": model_load_seconds,
                "memory_after_4bit_load": memory_after_4bit_load,
                "quantization": quantization,
                "lora": lora,
            },
            ensure_ascii=False,
        )
    )

    trainer_output = Path("/tmp/execsql_qlora_smoke_trainer")
    training_args = TrainingArguments(
        output_dir=str(trainer_output),
        max_steps=args.max_steps,
        num_train_epochs=3.0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        push_to_hub=False,
        logging_nan_inf_filter=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SingleTurnDataset(smoke_turn),
        data_collator=AssistantOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    train_started = perf_counter()
    try:
        train_output = trainer.train()
    except torch.OutOfMemoryError:
        print(json.dumps({"stage": "training_failed", "oom": True}))
        raise
    step_wall_seconds = perf_counter() - train_started
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    if trainer.state.global_step != 1:
        raise ValueError(f"Expected global_step=1, found {trainer.state.global_step}")
    step_log = find_step_log(trainer.state.log_history)
    training_loss = float(step_log["loss"])
    if not math.isfinite(training_loss) or not math.isfinite(train_output.training_loss):
        raise ValueError(
            f"Non-finite training loss: log={training_loss}, output={train_output.training_loss}"
        )
    deltas = lora_deltas(model, lora_before)
    adapter = save_adapter(model=model, tokenizer=tokenizer, output_path=args.output)

    training_report = {
        "stage": "training_complete",
        "optimizer": "paged_adamw_8bit",
        "global_step": trainer.state.global_step,
        "training_loss": training_loss,
        "train_output_loss": train_output.training_loss,
        "learning_rate": step_log.get("learning_rate"),
        "grad_norm": step_log.get("grad_norm"),
        "train_runtime": train_output.metrics.get("train_runtime"),
        "step_wall_seconds": step_wall_seconds,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "lora_max_abs_delta": deltas,
        "adapter": adapter,
        "oom": False,
        "nan_or_inf": False,
    }
    print(json.dumps(training_report, ensure_ascii=False))

    del trainer, train_output, model
    gc.collect()
    torch.cuda.empty_cache()
    reload_loss = reload_adapter_forward(
        model_path=args.model,
        adapter_path=args.output,
        turn=smoke_turn,
    )
    final_report = {
        "stage": "reload_complete",
        "adapter_reload_success": True,
        "reload_forward_loss": reload_loss,
        "final_cuda_memory": cuda_memory(),
    }
    print(json.dumps(final_report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
