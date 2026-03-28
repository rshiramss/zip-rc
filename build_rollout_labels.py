"""Step 2A: create per-prefix labels from one completed rollout.

This script does the minimal first pass described in the paper:
1. take a prompt
2. generate one full answer (or accept a provided completion)
3. score the final answer once
4. walk each non-empty prefix of the completion
5. save one label per prefix:
   - reward bin: 0 = wrong, 1 = correct
   - remaining-length bin:
       0 = 0-9 tokens left
       1 = 10-19
       2 = 20-39
       3 = 40-79
       4 = 80+

The output is JSONL, one row per prefix, with an additional ``grid_cell`` field
that maps the reward/length pair to a single class id.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NUM_REWARD_BINS = 2
NUM_LENGTH_BINS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Prompt to evaluate.")
    parser.add_argument(
        "--expected-answer",
        required=True,
        help="Reference answer used by the baseline scorer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rollout_labels.jsonl"),
        help="Where to write JSONL prefix labels.",
    )
    parser.add_argument(
        "--completion",
        help="Use this completion directly instead of generating one from a model.",
    )
    parser.add_argument(
        "--model",
        help="Model name or path for generation/tokenization when --completion is not supplied.",
    )
    parser.add_argument(
        "--tokenizer",
        help="Optional tokenizer name/path. Defaults to --model when present.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Generation cap when using --model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device used for generation when --model is supplied.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible generation.",
    )
    return parser.parse_args()


def normalize_for_scoring(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.?!,;:]+$", "", text)
    return text


def is_correct_completion(completion: str, expected_answer: str) -> bool:
    """Simple baseline scorer.

    Marks a completion as correct when its normalized token suffix matches the
    normalized reference answer. This keeps the example
    "2 + 2 = 4" correct for the expected answer "4".
    """

    normalized_completion = normalize_for_scoring(completion)
    normalized_expected = normalize_for_scoring(expected_answer)

    if not normalized_expected:
        raise ValueError("Expected answer is empty after normalization.")

    completion_tokens = normalized_completion.split()
    expected_tokens = normalized_expected.split()

    if len(completion_tokens) < len(expected_tokens):
        return False

    return completion_tokens[-len(expected_tokens) :] == expected_tokens


def reward_bin_for_completion(completion: str, expected_answer: str) -> int:
    return int(is_correct_completion(completion, expected_answer))


def length_bin_for_remaining_tokens(tokens_left: int) -> int:
    if tokens_left <= 9:
        return 0
    if tokens_left <= 19:
        return 1
    if tokens_left <= 39:
        return 2
    if tokens_left <= 79:
        return 3
    return 4


def grid_cell_id(reward_bin: int, length_bin: int) -> int:
    return reward_bin * NUM_LENGTH_BINS + length_bin


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(tokenizer_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_completion(
    prompt: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str | None,
    max_new_tokens: int,
    device: str,
    seed: int,
) -> tuple[str, list[int], Any]:
    import torch
    from transformers import AutoModelForCausalLM, set_seed

    tokenizer = load_tokenizer(tokenizer_name_or_path or model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    target_device = resolve_device(device)
    model.to(target_device)
    model.eval()
    set_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(target_device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0]

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[prompt_length:].tolist()
    generated_ids = trim_trailing_special_ids(generated_ids, tokenizer)
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return completion, generated_ids, tokenizer


def trim_trailing_special_ids(token_ids: list[int], tokenizer: Any) -> list[int]:
    special_ids = set(tokenizer.all_special_ids)
    trimmed = list(token_ids)
    while trimmed and trimmed[-1] in special_ids:
        trimmed.pop()
    return trimmed


def whitespace_tokenize(text: str) -> list[str]:
    return text.split()


def build_prefix_rows_from_token_ids(
    prompt: str,
    completion: str,
    token_ids: list[int],
    tokenizer: Any,
    reward_bin: int,
) -> list[dict[str, Any]]:
    total_tokens = len(token_ids)
    rows: list[dict[str, Any]] = []

    for prefix_length in range(1, total_tokens + 1):
        prefix_ids = token_ids[:prefix_length]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        tokens_left = total_tokens - prefix_length
        length_bin = length_bin_for_remaining_tokens(tokens_left)

        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "prefix_text": prefix_text,
                "prefix_token_ids": prefix_ids,
                "prefix_length_tokens": prefix_length,
                "remaining_tokens": tokens_left,
                "reward_bin": reward_bin,
                "remaining_length_bin": length_bin,
                "grid_cell": grid_cell_id(reward_bin, length_bin),
            }
        )

    return rows


def build_prefix_rows_from_whitespace(
    prompt: str,
    completion: str,
    reward_bin: int,
) -> list[dict[str, Any]]:
    tokens = whitespace_tokenize(completion)
    rows: list[dict[str, Any]] = []
    total_tokens = len(tokens)

    for prefix_length in range(1, total_tokens + 1):
        prefix_tokens = tokens[:prefix_length]
        tokens_left = total_tokens - prefix_length
        length_bin = length_bin_for_remaining_tokens(tokens_left)

        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "prefix_text": " ".join(prefix_tokens),
                "prefix_tokens": prefix_tokens,
                "prefix_length_tokens": prefix_length,
                "remaining_tokens": tokens_left,
                "reward_bin": reward_bin,
                "remaining_length_bin": length_bin,
                "grid_cell": grid_cell_id(reward_bin, length_bin),
            }
        )

    return rows


def save_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()

    if args.completion is None and args.model is None:
        raise ValueError("Pass either --completion or --model.")

    tokenizer = None
    token_ids: list[int] | None = None

    if args.completion is not None:
        completion = args.completion
        tokenizer_name_or_path = args.tokenizer or args.model
        if tokenizer_name_or_path:
            tokenizer = load_tokenizer(tokenizer_name_or_path)
            token_ids = tokenizer.encode(completion, add_special_tokens=False)
    else:
        completion, token_ids, tokenizer = generate_completion(
            prompt=args.prompt,
            model_name_or_path=args.model,
            tokenizer_name_or_path=args.tokenizer,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            seed=args.seed,
        )

    if not completion.strip():
        raise ValueError("Completion is empty after generation/tokenization.")

    reward_bin = reward_bin_for_completion(completion, args.expected_answer)

    if tokenizer is not None and token_ids is not None:
        rows = build_prefix_rows_from_token_ids(
            prompt=args.prompt,
            completion=completion,
            token_ids=token_ids,
            tokenizer=tokenizer,
            reward_bin=reward_bin,
        )
        tokenization_mode = "model_tokenizer"
    else:
        rows = build_prefix_rows_from_whitespace(
            prompt=args.prompt,
            completion=completion,
            reward_bin=reward_bin,
        )
        tokenization_mode = "whitespace_fallback"

    save_rows(rows, args.output)

    summary = {
        "prompt": args.prompt,
        "completion": completion,
        "reward_bin": reward_bin,
        "num_prefixes": len(rows),
        "tokenization_mode": tokenization_mode,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
