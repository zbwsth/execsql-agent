"""Validate assistant-turn-only preprocessing for Qwen3 tool-calling SFT.

This module deliberately loads only the tokenizer. It does not load model weights,
create a Trainer, or modify the source JSONL files.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass(frozen=True)
class ProcessedTurn:
    """One assistant message converted to prompt-masked causal-LM tensors."""

    case_id: str
    split: str
    template_family: str
    turn_index: int
    target_type: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_tokens_before_trim: int
    target_tokens: int
    original_sequence_tokens: int
    left_trimmed_prompt_tokens: int
    target_decode: str
    tool_call_has_loss: bool
    final_answer_has_loss: bool
    target_has_turn_end: bool
    context_tokens_supervised: int

    @property
    def sequence_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def prompt_masked_tokens(self) -> int:
        return sum(label == -100 for label in self.labels)

    @property
    def supervised_tokens(self) -> int:
        return sum(label != -100 for label in self.labels)


class PrefixMismatchError(ValueError):
    """Raised when a completed conversation does not extend its prompt exactly."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read independent JSON objects without modifying the source file."""

    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            samples.append(raw)
    return samples


def _target_type(message: dict[str, Any]) -> str:
    if message.get("tool_calls"):
        return "tool_call"
    content = message.get("content")
    if isinstance(content, str) and content:
        return "final_answer"
    raise ValueError("Assistant message has neither tool_calls nor final content")


def _encode_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if not isinstance(input_ids, list) or not all(
        isinstance(token_id, int) for token_id in input_ids
    ):
        raise TypeError("Tokenizer did not return list[int]")
    return input_ids


def process_assistant_turn(
    *,
    tokenizer: PreTrainedTokenizerBase,
    sample: dict[str, Any],
    split: str,
    assistant_index: int,
    max_length: int,
) -> ProcessedTurn:
    """Build one assistant-only sample using strict prompt/completion prefixing."""

    messages = sample.get("messages")
    tools = sample.get("tools")
    metadata = sample.get("metadata")
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise ValueError("Sample requires messages and tools arrays")
    if not isinstance(metadata, dict):
        raise ValueError("Sample requires metadata")
    assistant_message = messages[assistant_index]
    if not isinstance(assistant_message, dict) or assistant_message.get("role") != "assistant":
        raise ValueError("assistant_index does not point to an assistant message")
    history = messages[:assistant_index]
    completed_messages = messages[: assistant_index + 1]

    prompt_text = tokenizer.apply_chat_template(
        history,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    completed_text = tokenizer.apply_chat_template(
        completed_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(prompt_text, str) or not isinstance(completed_text, str):
        raise TypeError("Chat template did not return text")

    prompt_ids = _encode_text(tokenizer, prompt_text)
    completed_ids = _encode_text(tokenizer, completed_text)
    if completed_ids[: len(prompt_ids)] != prompt_ids:
        common = 0
        for prompt_token, completed_token in zip(prompt_ids, completed_ids, strict=False):
            if prompt_token != completed_token:
                break
            common += 1
        boundary_start = max(0, len(prompt_text) - 80)
        boundary_end = len(prompt_text) + 160
        completed_boundary = completed_text[boundary_start:boundary_end]
        raise PrefixMismatchError(
            f"{metadata.get('case_id')} assistant turn {assistant_index}: "
            f"prompt_tokens={len(prompt_ids)}, completed_tokens={len(completed_ids)}, "
            f"common_prefix_tokens={common}, "
            f"prompt_tail={prompt_text[-160:]!r}, "
            f"completed_boundary={completed_boundary!r}"
        )
    target_ids = completed_ids[len(prompt_ids) :]
    if not target_ids:
        raise ValueError(
            f"{metadata.get('case_id')} assistant turn {assistant_index}: empty target"
        )
    if len(target_ids) >= max_length:
        raise ValueError(
            f"{metadata.get('case_id')} assistant turn {assistant_index}: "
            f"target length {len(target_ids)} is not smaller than {max_length}"
        )

    original_sequence_tokens = len(completed_ids)
    left_trimmed = max(0, original_sequence_tokens - max_length)
    if left_trimmed > len(prompt_ids):
        raise ValueError(
            f"{metadata.get('case_id')} assistant turn {assistant_index}: target truncation"
        )
    retained_prompt_tokens = len(prompt_ids) - left_trimmed
    input_ids = completed_ids[left_trimmed:]
    labels = [-100] * retained_prompt_tokens + target_ids
    attention_mask = [1] * len(input_ids)
    if len(input_ids) != len(labels) or len(labels) != len(attention_mask):
        raise ValueError("input_ids, labels and attention_mask lengths differ")
    if labels[retained_prompt_tokens:] != input_ids[retained_prompt_tokens:]:
        raise ValueError("Assistant target labels differ from input_ids")
    context_tokens_supervised = sum(label != -100 for label in labels[:retained_prompt_tokens])

    target_type = _target_type(assistant_message)
    target_decode = tokenizer.decode(target_ids, skip_special_tokens=False)
    supervised_count = len(target_ids)
    tool_call_has_loss = (
        target_type == "tool_call" and supervised_count > 0 and "<tool_call>" in target_decode
    )
    final_content = assistant_message.get("content")
    final_answer_has_loss = (
        target_type == "final_answer"
        and supervised_count > 0
        and isinstance(final_content, str)
        and final_content in target_decode
    )
    turn_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    target_has_turn_end = isinstance(turn_end_token_id, int) and turn_end_token_id in target_ids
    return ProcessedTurn(
        case_id=str(metadata.get("case_id")),
        split=split,
        template_family=str(metadata.get("template_family")),
        turn_index=assistant_index,
        target_type=target_type,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_tokens_before_trim=len(prompt_ids),
        target_tokens=len(target_ids),
        original_sequence_tokens=original_sequence_tokens,
        left_trimmed_prompt_tokens=left_trimmed,
        target_decode=target_decode,
        tool_call_has_loss=tool_call_has_loss,
        final_answer_has_loss=final_answer_has_loss,
        target_has_turn_end=target_has_turn_end,
        context_tokens_supervised=context_tokens_supervised,
    )


def process_trajectory(
    *,
    tokenizer: PreTrainedTokenizerBase,
    sample: dict[str, Any],
    split: str,
    max_length: int,
) -> list[ProcessedTurn]:
    """Split every assistant message in one trajectory into its own sample."""

    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages is not an array")
    turns: list[ProcessedTurn] = []
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            turns.append(
                process_assistant_turn(
                    tokenizer=tokenizer,
                    sample=sample,
                    split=split,
                    assistant_index=index,
                    max_length=max_length,
                )
            )
    if not turns:
        raise ValueError("Trajectory has no assistant messages")
    return turns


def _stats(values: list[int]) -> dict[str, int | float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _turn_summary(turn: ProcessedTurn) -> dict[str, object]:
    prefix_chars = 180
    suffix_chars = 180
    return {
        "turn_index": turn.turn_index,
        "type": turn.target_type,
        "total_tokens": turn.sequence_tokens,
        "prompt_masked_tokens": turn.prompt_masked_tokens,
        "supervised_tokens": turn.supervised_tokens,
        "supervised_ratio": turn.supervised_tokens / turn.sequence_tokens,
        "target_decode_prefix": turn.target_decode[:prefix_chars],
        "target_decode_suffix": turn.target_decode[-suffix_chars:],
        "tool_call_has_supervised_token": turn.tool_call_has_loss,
        "final_answer_has_supervised_token": turn.final_answer_has_loss,
        "target_has_turn_end_token": turn.target_has_turn_end,
        "context_tokens_supervised": turn.context_tokens_supervised,
        "left_trimmed_prompt_tokens": turn.left_trimmed_prompt_tokens,
    }


def validate_all(
    *,
    tokenizer: PreTrainedTokenizerBase,
    datasets: dict[str, list[dict[str, Any]]],
    max_length: int,
    selected_case_ids: set[str] | None = None,
) -> dict[str, object]:
    """Run full train/dev validation and enforce every hard-stop condition."""

    selected_case_ids = selected_case_ids or set()
    turns_by_split: dict[str, list[ProcessedTurn]] = defaultdict(list)
    selected: dict[str, dict[str, object]] = {}
    prefix_mismatches: list[str] = []
    failures: list[str] = []

    for split, trajectories in datasets.items():
        for sample in trajectories:
            metadata = sample.get("metadata")
            case_id = metadata.get("case_id") if isinstance(metadata, dict) else None
            try:
                turns = process_trajectory(
                    tokenizer=tokenizer,
                    sample=sample,
                    split=split,
                    max_length=max_length,
                )
                turns_by_split[split].extend(turns)
                if case_id in selected_case_ids:
                    selected[str(case_id)] = {
                        "original_assistant_turns": len(turns),
                        "training_samples": len(turns),
                        "turns": [_turn_summary(turn) for turn in turns],
                    }
            except PrefixMismatchError as error:
                prefix_mismatches.append(str(error))
            except Exception as error:
                failures.append(f"{case_id}: {type(error).__name__}: {error}")

    all_turns = [turn for turns in turns_by_split.values() for turn in turns]
    zero_supervised = sum(turn.supervised_tokens == 0 for turn in all_turns)
    tool_targets = [turn for turn in all_turns if turn.target_type == "tool_call"]
    final_targets = [turn for turn in all_turns if turn.target_type == "final_answer"]
    tool_without_loss = sum(not turn.tool_call_has_loss for turn in tool_targets)
    final_without_loss = sum(not turn.final_answer_has_loss for turn in final_targets)
    context_supervision_violations = sum(turn.context_tokens_supervised > 0 for turn in all_turns)
    missing_turn_end = sum(not turn.target_has_turn_end for turn in all_turns)
    target_truncations = sum(
        turn.left_trimmed_prompt_tokens > turn.prompt_tokens_before_trim for turn in all_turns
    )

    split_reports: dict[str, object] = {}
    for split, trajectories in datasets.items():
        turns = turns_by_split[split]
        supervised = [turn.supervised_tokens for turn in turns]
        sequences = [turn.sequence_tokens for turn in turns]
        split_reports[split] = {
            "trajectories": len(trajectories),
            "assistant_turn_samples": len(turns),
            "tool_call_targets": sum(turn.target_type == "tool_call" for turn in turns),
            "final_answer_targets": sum(turn.target_type == "final_answer" for turn in turns),
            "supervised_tokens": _stats(supervised) if supervised else None,
            "sequence_lengths": _stats(sequences) if sequences else None,
            "zero_supervised_samples": sum(turn.supervised_tokens == 0 for turn in turns),
            "sequences_over_max_length_before_prompt_trim": sum(
                turn.original_sequence_tokens > max_length for turn in turns
            ),
            "prompt_left_trimmed_samples": sum(
                turn.left_trimmed_prompt_tokens > 0 for turn in turns
            ),
        }

    hard_stop = bool(
        prefix_mismatches
        or failures
        or zero_supervised
        or tool_without_loss
        or final_without_loss
        or context_supervision_violations
        or missing_turn_end
        or target_truncations
    )
    return {
        "max_length": max_length,
        "enable_thinking": False,
        "selected_cases": selected,
        "splits": split_reports,
        "validation": {
            "zero_supervised_samples": zero_supervised,
            "prefix_mismatch_count": len(prefix_mismatches),
            "tool_call_without_loss_count": tool_without_loss,
            "final_answer_without_loss_count": final_without_loss,
            "context_supervision_violation_count": context_supervision_violations,
            "missing_assistant_turn_end_count": missing_turn_end,
            "target_truncation_count": target_truncations,
            "other_failure_count": len(failures),
            "hard_stop": hard_stop,
            "prefix_mismatch_examples": prefix_mismatches[:5],
            "failure_examples": failures[:5],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Qwen3 assistant-turn-only tool-calling preprocessing."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--inspect-case-id",
        action="append",
        default=[],
        help="Include detailed turn summaries for this case id; repeat as needed.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_length < 1:
        raise ValueError("max_length must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    report = validate_all(
        tokenizer=tokenizer,
        datasets={"train": load_jsonl(args.train), "dev": load_jsonl(args.dev)},
        max_length=args.max_length,
        selected_case_ids=set(args.inspect_case_id),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["validation"]["hard_stop"])


if __name__ == "__main__":
    raise SystemExit(main())
