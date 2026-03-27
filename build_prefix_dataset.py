"""
Build the prefix-labeled training dataset for ZIP-RC.

For each scored rollout with T completion tokens:
  - Walk prefixes t=1..T
  - Compute tokens_left = T - t
  - Map reward → reward_bin, tokens_left → length_bin
  - Compute joint_label = reward_bin * BT + length_bin

Reads data/scored_rollouts.jsonl, writes data/prefix_dataset.jsonl.
"""

import json
from transformers import AutoTokenizer
from zip_rc_model import DEFAULT_BV, DEFAULT_BT, tokens_left_to_bin, joint_label

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
INPUT_PATH = "data/scored_rollouts.jsonl"
OUTPUT_PATH = "data/prefix_dataset.jsonl"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open(INPUT_PATH) as f:
        rollouts = [json.loads(line) for line in f]

    examples = []
    for row in rollouts:
        # Reconstruct full input: prompt + completion tokens
        prompt_text = f"Q: {row['prompt']}\nA:"
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
                "reward_bin": reward_bin,
                "length_bin": length_bin,
                "joint_label": label,
            })

    with open(OUTPUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Built {len(examples)} prefix examples from {len(rollouts)} rollouts")
    print(f"Joint labels range: 0 to {DEFAULT_BV * DEFAULT_BT - 1}")
    print(f"Wrote to {OUTPUT_PATH}")

    # Quick sanity check
    labels = [ex["joint_label"] for ex in examples]
    from collections import Counter
    dist = Counter(labels)
    print(f"Label distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
