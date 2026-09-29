"""Full QLoRA SFT for ExecSQL-Agent.

Reuses the already smoke-tested QLoRA/model/collator helpers and the
validated assistant-only preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from assistant_turn_preprocessing import (
    ProcessedTurn,
    load_jsonl,
    process_trajectory,
)
from torch.utils.data import Dataset
from train_qlora_sft import (
    AssistantOnlyCollator,
    attach_lora,
    cuda_memory,
    inspect_quantization,
    load_quantized_base,
    optimizer_supported,
)
from transformers import AutoTokenizer, Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]


MAX_LENGTH = 4096


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for one provenance input."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, object]:
    """Return a stable fingerprint for one required local file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_revision(path: Path) -> str | None:
    """Return the Git revision containing path when metadata is available."""

    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    revision = result.stdout.strip()
    return revision or None


def package_version(name: str) -> str | None:
    """Return an installed distribution version without importing it."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist one human-readable JSON artifact."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def environment_snapshot() -> dict[str, object]:
    """Capture the software environment used by the completed training run."""

    package_names = (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name) for name in package_names
        },
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": cuda_memory() if torch.cuda.is_available() else None,
    }


class AssistantTurnDataset(Dataset[dict[str, list[int]]]):
    """Dataset of already-tokenized assistant-only training turns."""

    def __init__(self, turns: list[ProcessedTurn]) -> None:
        self.turns = turns

    def __len__(self) -> int:
        return len(self.turns)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        turn = self.turns[index]
        return {
            "input_ids": turn.input_ids,
            "attention_mask": turn.attention_mask,
            "labels": turn.labels,
        }


def load_turns(
    *,
    tokenizer,
    path: Path,
    split: str,
    max_length: int,
) -> tuple[list[ProcessedTurn], int]:
    """Convert every trajectory into assistant-turn training samples."""

    trajectories = load_jsonl(path)
    turns: list[ProcessedTurn] = []

    for sample in trajectories:
        turns.extend(
            process_trajectory(
                tokenizer=tokenizer,
                sample=sample,
                split=split,
                max_length=max_length,
            )
        )

    if not turns:
        raise ValueError(f"{split} produced zero assistant-turn samples")

    return turns, len(trajectories)


def validate_turns(
    *,
    turns: list[ProcessedTurn],
    split: str,
) -> dict[str, object]:
    """Hard-stop validation before touching the 8B model."""

    zero_supervised = [
        turn.case_id for turn in turns if turn.supervised_tokens == 0
    ]
    context_supervised = [
        turn.case_id
        for turn in turns
        if turn.context_tokens_supervised != 0
    ]
    missing_turn_end = [
        turn.case_id
        for turn in turns
        if not turn.target_has_turn_end
    ]
    too_long = [
        turn.case_id
        for turn in turns
        if turn.sequence_tokens > MAX_LENGTH
    ]

    if zero_supervised:
        raise ValueError(
            f"{split}: zero-supervised samples: {zero_supervised[:5]}"
        )
    if context_supervised:
        raise ValueError(
            f"{split}: context tokens unexpectedly supervised: "
            f"{context_supervised[:5]}"
        )
    if missing_turn_end:
        raise ValueError(
            f"{split}: assistant targets missing turn-end token: "
            f"{missing_turn_end[:5]}"
        )
    if too_long:
        raise ValueError(
            f"{split}: sequence exceeds {MAX_LENGTH}: {too_long[:5]}"
        )

    lengths = [turn.sequence_tokens for turn in turns]
    supervised = [turn.supervised_tokens for turn in turns]

    return {
        "samples": len(turns),
        "sequence_min": min(lengths),
        "sequence_mean": sum(lengths) / len(lengths),
        "sequence_max": max(lengths),
        "supervised_min": min(supervised),
        "supervised_mean": sum(supervised) / len(supervised),
        "supervised_max": max(supervised),
        "tool_call_targets": sum(
            turn.target_type == "tool_call" for turn in turns
        ),
        "final_answer_targets": sum(
            turn.target_type == "final_answer" for turn in turns
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full ExecSQL-Agent Qwen3-8B QLoRA SFT."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output}"
        )

    input_fingerprints = {
        "train": file_fingerprint(args.train),
        "dev": file_fingerprint(args.dev),
        "tools": file_fingerprint(args.tools),
    }
    model_files = [
        args.model / "config.json",
        args.model / "tokenizer_config.json",
        args.model / "model.safetensors.index.json",
        *sorted(args.model.glob("model-*.safetensors")),
    ]
    model_fingerprints = {
        path.name: file_fingerprint(path) for path in model_files
    }
    project_revision = git_revision(PROJECT_ROOT)
    model_revision = git_revision(args.model)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError(
            "--gradient-accumulation-steps must be positive"
        )

    optimizer_supported()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id"
            )
        tokenizer.pad_token = tokenizer.eos_token

    train_turns, train_trajectory_count = load_turns(
        tokenizer=tokenizer,
        path=args.train,
        split="train",
        max_length=MAX_LENGTH,
    )
    dev_turns, dev_trajectory_count = load_turns(
        tokenizer=tokenizer,
        path=args.dev,
        split="dev",
        max_length=MAX_LENGTH,
    )

    train_stats = validate_turns(turns=train_turns, split="train")
    dev_stats = validate_turns(turns=dev_turns, split="dev")

    print(
        json.dumps(
            {
                "stage": "data_ready",
                "train_trajectories": train_trajectory_count,
                "dev_trajectories": dev_trajectory_count,
                "train": train_stats,
                "dev": dev_stats,
            },
            ensure_ascii=False,
        )
    )

    torch.cuda.empty_cache()

    load_started = perf_counter()
    model = load_quantized_base(args.model)
    model_load_seconds = perf_counter() - load_started

    memory_after_4bit_load = cuda_memory()
    quantization = inspect_quantization(model)

    model, lora = attach_lora(model)

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

    trainer_output = Path("/tmp/execsql_qlora_full_trainer")

    training_args = TrainingArguments(
        output_dir=str(trainer_output),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        max_grad_norm=1.0,

        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,

        eval_strategy="epoch",
        save_strategy="no",
        label_names=["labels"],

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
        train_dataset=AssistantTurnDataset(train_turns),
        eval_dataset=AssistantTurnDataset(dev_turns),
        data_collator=AssistantOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    train_started = perf_counter()

    try:
        train_output = trainer.train()
    except torch.OutOfMemoryError:
        print(
            json.dumps(
                {
                    "stage": "training_failed",
                    "oom": True,
                }
            )
        )
        raise

    wall_seconds = perf_counter() - train_started

    train_loss = float(train_output.training_loss)

    if not math.isfinite(train_loss):
        raise ValueError(
            f"Final training loss is not finite: {train_loss}"
        )

    final_eval = trainer.evaluate()
    eval_loss = float(final_eval["eval_loss"])

    if not math.isfinite(eval_loss):
        raise ValueError(
            f"Final dev loss is not finite: {eval_loss}"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    # Save PEFT adapter + tokenizer only; never save the 8B base model.
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    trainer.state.save_to_json(str(args.output / "trainer_state.json"))

    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)

    expected_optimizer_steps = math.ceil(
        len(train_turns) / args.gradient_accumulation_steps
    ) * int(args.epochs)

    training_configuration = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "max_length": MAX_LENGTH,
        "bf16": True,
        "gradient_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
        "assistant_only_supervision": True,
        "quantization": {
            "load_in_4bit": True,
            "type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quantization": True,
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": "all-linear",
            "task_type": "CAUSAL_LM",
        },
    }
    training_summary = {
        "completed_at": datetime.now(UTC).isoformat(),
        "global_step": trainer.state.global_step,
        "expected_optimizer_steps": expected_optimizer_steps,
        "training_loss": train_loss,
        "final_eval_loss": eval_loss,
        "train_runtime_wall_seconds": wall_seconds,
        "peak_allocated_gib": peak_allocated / (1024**3),
        "peak_reserved_gib": peak_reserved / (1024**3),
        "train_trajectories": train_trajectory_count,
        "train_assistant_turns": len(train_turns),
        "dev_trajectories": dev_trajectory_count,
        "dev_assistant_turns": len(dev_turns),
        "train_statistics": train_stats,
        "dev_statistics": dev_stats,
        "configuration": training_configuration,
    }
    training_manifest = {
        "manifest_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "project_git_revision": project_revision,
        "training_script": file_fingerprint(Path(__file__)),
        "base_model": {
            "path": str(args.model.resolve()),
            "git_revision": model_revision,
            "files": model_fingerprints,
        },
        "inputs": input_fingerprints,
        "output": str(args.output.resolve()),
        "configuration": training_configuration,
    }
    write_json(args.output / "training_summary.json", training_summary)
    write_json(args.output / "training_manifest.json", training_manifest)
    write_json(args.output / "environment.json", environment_snapshot())

    required_outputs = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_summary.json",
        "training_manifest.json",
        "environment.json",
    )
    missing_outputs = [
        name for name in required_outputs if not (args.output / name).is_file()
    ]
    if missing_outputs:
        raise FileNotFoundError(
            f"Training output is incomplete: {missing_outputs}"
        )

    print(
        json.dumps(
            {
                "stage": "training_complete",
                "global_step": trainer.state.global_step,
                "expected_optimizer_steps": expected_optimizer_steps,
                "training_loss": train_loss,
                "final_eval_loss": eval_loss,
                "train_runtime_wall_seconds": wall_seconds,
                "peak_allocated_gib": peak_allocated / (1024**3),
                "peak_reserved_gib": peak_reserved / (1024**3),
                "output": str(args.output),
                "output_files": sorted(
                    path.name
                    for path in args.output.iterdir()
                    if path.is_file()
                ),
            },
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
