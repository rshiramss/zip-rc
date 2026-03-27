"""
Generate one full completion per prompt using the base model (no ZIP tokens).
Reads data/prompts.jsonl, writes data/rollouts.jsonl.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS_PATH = "data/prompts.jsonl"
OUTPUT_PATH = "data/rollouts.jsonl"
MAX_NEW_TOKENS = 256


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(PROMPTS_PATH) as f:
        prompts = [json.loads(line) for line in f]

    results = []
    for i, row in enumerate(prompts):
        # Format as a simple instruction
        text = f"Q: {row['prompt']}\nA:"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id,
            )

        completion_ids = output_ids[0, prompt_len:].tolist()
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        results.append({
            "prompt": row["prompt"],
            "answer": row["answer"],
            "completion": completion.strip(),
            "completion_ids": completion_ids,
        })

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{len(prompts)}")

    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(results)} rollouts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
