"""
Build prefix-labeled training dataset for ZIP-RC from rollout files.

For each rollout with T completion tokens, walks prefixes t=1..T and computes:
  - tokens_left = T - t
  - reward_bin (0=wrong, 1=correct)
  - length_bin (logarithmic buckets: 0-9, 10-19, 20-39, 40-79, 80+)
  - joint_label = reward_bin * BT + length_bin

Example usage:
    python build_prefix_dataset.py
    python build_prefix_dataset.py --input data/rollouts.jsonl --output data/prefix_dataset.jsonl
"""

import argparse
import json
from collections import Counter

from transformers import AutoTokenizer
from zip_rc_model import DEFAULT_BT, tokens_left_to_bin, joint_label


def resolve_tokenizer_model(rollouts, cli_model: str | None) -> str:
    """Resolve the tokenizer model from rollout metadata or an explicit override."""
    rollout_models = {
        row["model_name"]
        for row in rollouts
        if row.get("model_name")
    }

    if len(rollout_models) > 1:
        raise ValueError(
            "Input rollouts contain multiple model_name values. "
            "Build one prefix dataset per base model."
        )

    rollout_model = next(iter(rollout_models), None)

    if cli_model is not None:
        if rollout_model is not None and cli_model != rollout_model:
            raise ValueError(
                f"Tokenizer mismatch: rollouts were generated with {rollout_model!r}, "
                f"but --model={cli_model!r} was requested."
            )
        return cli_model

    if rollout_model is not None:
        return rollout_model

    raise ValueError(
        "Could not determine the tokenizer model from the rollout file. "
        "Pass --model explicitly or regenerate rollouts with the current "
        "generate_rollouts.py so model_name metadata is recorded."
    )


def main():
    parser = argparse.ArgumentParser(description="Build ZIP-RC prefix dataset from rollouts")
    parser.add_argument("--model", type=str, default=None,
                        help=("Model name used for tokenization. Defaults to the model_name "
                              "stored in the rollout file."))
    parser.add_argument("--input", type=str, default="data/rollouts.jsonl",
                        help="Input rollouts JSONL file")
    parser.add_argument("--output", type=str, default="data/prefix_dataset.jsonl",
                        help="Output prefix dataset JSONL file")
    args = parser.parse_args()

    # Read rollouts
    with open(args.input) as f:
        rollouts = [json.loads(line) for line in f]
    print(f"Loaded {len(rollouts)} rollouts from {args.input}")

    tokenizer_model = resolve_tokenizer_model(rollouts, args.model)
    print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    examples = []
    for row in rollouts:
        question = row.get("question", row.get("prompt"))
        if question is None:
            raise KeyError(
                "Rollout row is missing both 'question' and 'prompt' fields. "
                "Unsupported rollout schema."
            )

        # Tokenize the prompt (question formatted as in generate_rollouts.py)
        prompt_text = f"Q: {question}\nA:"
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        completion_ids = row["completion_ids"]
        T = len(completion_ids)

        if T == 0:
            continue

        reward_bin = row["reward"]  # already 0 or 1

        # Walk every prefix of the completion
        for t in range(1, T + 1):
            tokens_left = T - t
            length_bin = tokens_left_to_bin(tokens_left)
            label = joint_label(reward_bin, length_bin, bt=DEFAULT_BT)

            input_ids = prompt_ids + completion_ids[:t]
            attention_mask = [1] * len(input_ids)

            examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "model_name": tokenizer_model,
                "joint_label": label,
                "reward_bin": reward_bin,
                "length_bin": length_bin,
            })

    # Write output
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nBuilt {len(examples)} prefix examples from {len(rollouts)} rollouts")
    print(f"Joint labels range: 0 to 9")
    print(f"Wrote to {args.output}")

    # Label distribution
    dist = Counter(ex["joint_label"] for ex in examples)
    print(f"\nLabel distribution:")
    for label in sorted(dist.keys()):
        rb, lb = divmod(label, DEFAULT_BT)
        print(f"  label {label} (reward={rb}, length={lb}): {dist[label]} examples")

    # Sanity check: print a few examples for manual inspection
    print(f"\n--- Sanity check (3 sampled examples) ---")
    import random
    check_indices = random.sample(range(len(examples)), min(3, len(examples)))
    for idx in check_indices:
        ex = examples[idx]
        prefix_len = len(ex["input_ids"])
        # Find the rollout this came from to show context
        decoded = tokenizer.decode(ex["input_ids"][-10:], skip_special_tokens=True)
        print(f"  Example {idx}:")
        print(f"    ...last tokens: \"{decoded}\"")
        print(f"    prefix_len={prefix_len}, reward_bin={ex['reward_bin']}, "
              f"length_bin={ex['length_bin']}, joint_label={ex['joint_label']}")


if __name__ == "__main__":
    main()
