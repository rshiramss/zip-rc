"""
ZIP-RC Model Wrapper

Takes a standard causal language model and extends it to output:
  (1) Normal token logits (original vocabulary)
  (2) ZIP logits (reserved slice for introspection: reward + length predictions)

Approach: add special <ZIP_0> ... <ZIP_N> tokens to the tokenizer, resize
the model embeddings, then split logits after each forward pass. The last
N logit dimensions correspond to ZIP tokens — used for reward/cost prediction.
Single forward pass, zero overhead.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


NUM_ZIP_TOKENS = 56  # 28 reward bins + 28 length bins (configurable)


class ZipRCModel(nn.Module):
    """Wraps a causal LM with reserved ZIP tokens for introspection.

    Step 1A: Adds <ZIP_0> .. <ZIP_{N-1}> to the tokenizer and resizes embeddings.
    Step 1B: On forward pass, splits logits into normal (text) and ZIP (introspection).
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_reward_bins: int = 28,
        num_length_bins: int = 28,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_reward_bins = num_reward_bins
        self.num_length_bins = num_length_bins
        self.num_zip_tokens = num_reward_bins + num_length_bins

        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        # Record original vocab size before adding ZIP tokens
        self.original_vocab_size = len(self.tokenizer)

        # Step 1A — Reserve ZIP tokens
        zip_tokens = [f"<ZIP_{i}>" for i in range(self.num_zip_tokens)]
        self.tokenizer.add_tokens(zip_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Freeze backbone if only training ZIP head
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze the new ZIP token embeddings + LM head
            # so they can be trained
            lm_head = self.model.lm_head
            for param in lm_head.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = outputs.logits  # (B, T, original_vocab + num_zip_tokens)

        # Step 1B — Split the logits
        normal_logits = logits[..., :self.original_vocab_size]
        zip_logits = logits[..., -self.num_zip_tokens:]

        # Step 1C — Mask ZIP tokens so they are NEVER generated as text
        # Read ZIP logits first, then set to -inf before any sampling
        reward_logits = zip_logits[..., :self.num_reward_bins]
        length_logits = zip_logits[..., self.num_reward_bins:]

        # Return generation-safe logits: ZIP positions masked to -inf
        generation_logits = logits.clone()
        generation_logits[..., -self.num_zip_tokens:] = float("-inf")

        return ZipRCOutput(
            token_logits=normal_logits,
            generation_logits=generation_logits,
            reward_logits=reward_logits,
            length_logits=length_logits,
        )

    def predict_zip(self, input_ids, attention_mask=None):
        """Return scalar expected reward and expected length from last position."""
        out = self.forward(input_ids, attention_mask)

        reward_probs = torch.softmax(out.reward_logits[:, -1, :], dim=-1)
        length_probs = torch.softmax(out.length_logits[:, -1, :], dim=-1)

        reward_bins = torch.linspace(0, 1, self.num_reward_bins, device=reward_probs.device)
        length_bins = torch.arange(self.num_length_bins, device=length_probs.device, dtype=torch.float)

        expected_reward = (reward_probs * reward_bins).sum(dim=-1)
        expected_length = (length_probs * length_bins).sum(dim=-1)

        return expected_reward, expected_length


class ZipRCOutput:
    """Container for ZIP-RC forward pass outputs."""

    __slots__ = ("token_logits", "generation_logits", "reward_logits", "length_logits")

    def __init__(self, token_logits, generation_logits, reward_logits, length_logits):
        self.token_logits = token_logits
        self.generation_logits = generation_logits
        self.reward_logits = reward_logits
        self.length_logits = length_logits


if __name__ == "__main__":
    model = ZipRCModel("gpt2", num_reward_bins=28, num_length_bins=28)
    tok = model.tokenizer
    inputs = tok("Hello, world!", return_tensors="pt")

    out = model(**inputs)
    print(f"Token logits shape:  {out.token_logits.shape}")   # (1, T, 50257)
    print(f"Reward logits shape: {out.reward_logits.shape}")   # (1, T, 28)
    print(f"Length logits shape: {out.length_logits.shape}")   # (1, T, 28)

    reward, length = model.predict_zip(**inputs)
    print(f"Expected reward: {reward.item():.3f}")
    print(f"Expected length: {length.item():.1f}")
