"""
Score rollouts by comparing the model's final numeric answer to ground truth.
Reads data/rollouts.jsonl, writes data/scored_rollouts.jsonl.
"""

import json
import re

INPUT_PATH = "data/rollouts.jsonl"
OUTPUT_PATH = "data/scored_rollouts.jsonl"


def extract_number(text: str) -> str | None:
    """Extract the last number from text (handles integers and decimals)."""
    matches = re.findall(r"-?\d+\.?\d*", text)
    if matches:
        return matches[-1]
    return None


def normalize_answer(ans: str) -> str:
    """Strip whitespace and trailing .0 for comparison."""
    ans = ans.strip()
    if ans.endswith(".0"):
        ans = ans[:-2]
    return ans


def main():
    with open(INPUT_PATH) as f:
        rollouts = [json.loads(line) for line in f]

    correct = 0
    results = []
    for row in rollouts:
        predicted = extract_number(row["completion"])
        expected = normalize_answer(row["answer"])

        if predicted is not None:
            predicted = normalize_answer(predicted)

        reward = 1 if predicted == expected else 0
        correct += reward

        results.append({**row, "predicted_answer": predicted, "reward": reward})

    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Scored {len(results)} rollouts: {correct}/{len(results)} correct ({100*correct/len(results):.1f}%)")
    print(f"Wrote to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
