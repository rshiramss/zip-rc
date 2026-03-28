"""
Generate rollouts from GSM8K using a causal language model.

For each question, generates one greedy completion, extracts the predicted
numeric answer, compares to ground truth, and assigns a binary reward.

Example usage:
    python generate_rollouts.py --num-examples 50
    python generate_rollouts.py --model Qwen/Qwen3-8B --num-examples 200 --output data/rollouts.jsonl
"""

import argparse
import json
import re
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from zip_rc_model import DEFAULT_MODEL_NAME


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(answer_field: str) -> str:
    """Extract the numeric answer from a GSM8K answer field.

    GSM8K answers contain reasoning followed by '#### <number>'.
    Returns the number string after '####', with commas removed.
    """
    match = re.search(r"####\s*(.+)", answer_field)
    if match:
        # Remove commas from numbers like "1,234"
        return match.group(1).strip().replace(",", "")
    return ""


def extract_number(text: str) -> str | None:
    """Extract the last number from text.

    Handles integers, decimals, and negative numbers.
    Returns None if no number is found.
    """
    # Remove commas within numbers (e.g., "1,234" -> "1234")
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", text)
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    if matches:
        # Normalize: strip trailing .0
        result = matches[-1]
        if result.endswith(".0"):
            result = result[:-2]
        return result
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate rollouts from GSM8K")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                        help="HuggingFace model name or path")
    parser.add_argument("--num-examples", type=int, default=50,
                        help="Number of GSM8K examples to process")
    parser.add_argument("--num-samples", type=int, default=2,
                        help="Number of sampled completions per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum tokens to generate per completion")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (ignored with --greedy)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of sampling")
    parser.add_argument("--output", type=str, default="data/rollouts.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")

    # Limit to requested number of examples
    n = min(args.num_examples, len(dataset))
    total_rollouts = n * args.num_samples
    print(f"Generating {args.num_samples} rollout(s) per prompt for {n} examples "
          f"({total_rollouts} total)...")

    # Generation config
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if args.greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature

    results = []
    correct_count = 0

    for i in range(n):
        example = dataset[i]
        question = example["question"]
        ground_truth = extract_gsm8k_answer(example["answer"])

        # Format prompt
        prompt = f"Q: {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        for s in range(args.num_samples):
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            completion_ids = output_ids[0, prompt_len:].tolist()
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # Score: extract predicted answer and compare to ground truth
            predicted_answer = extract_number(completion)
            reward = 1 if predicted_answer is not None and predicted_answer == ground_truth else 0
            correct_count += reward

            results.append({
                "model_name": args.model,
                "question": question,
                "completion": completion,
                "completion_ids": completion_ids,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "reward": reward,
            })

        if (i + 1) % 10 == 0:
            done = (i + 1) * args.num_samples
            print(f"  [{i + 1}/{n}] rollouts: {done}, correct: {correct_count}/{done} "
                  f"({100 * correct_count / done:.1f}%)")

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {correct_count}/{len(results)} correct ({100 * correct_count / len(results):.1f}%)")
    print(f"Wrote {len(results)} rollouts to {args.output}")


if __name__ == "__main__":
    main()
