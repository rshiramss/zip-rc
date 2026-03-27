"""
ZIP-RC Model Wrapper

Adds reserved ZIP tokens to a causal LM for joint (reward, length) prediction.
Each ZIP token maps to one cell in a BV x BT grid:
  - BV = number of reward bins (default 2: wrong/correct)
  - BT = number of length bins (default 5: logarithmic remaining-length buckets)
  - Total ZIP tokens = BV * BT

Single forward pass produces normal token logits + ZIP logits.
ZIP tokens are masked to -inf during generation so they're never sampled as text.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default bin configuration
DEFAULT_BV = 2   # reward bins: 0=wrong, 1=correct
DEFAULT_BT = 5   # length bins: 0-1, 2-3, 4-7, 8-15, 16+

# Length bin boundaries (upper bounds, inclusive). Last bin is open-ended.
LENGTH_BIN_EDGES = [1, 3, 7, 15]  # bin 0: 0-1, bin 1: 2-3, bin 2: 4-7, bin 3: 8-15, bin 4: 16+


def tokens_left_to_bin(tokens_left: int) -> int:
    """Map remaining token count to a length bin index."""
    for i, edge in enumerate(LENGTH_BIN_EDGES):
        if tokens_left <= edge:
            return i
    return len(LENGTH_BIN_EDGES)  # last open-ended bin


def joint_label(reward_bin: int, length_bin: int, bt: int = DEFAULT_BT) -> int:
    """Flatten (reward_bin, length_bin) into a single class index."""
    return reward_bin * bt + length_bin


class ZipRCModel(nn.Module):
    """Causal LM with reserved ZIP tokens for joint (reward, length) prediction.

    Step 1A: Adds <ZIP_0> .. <ZIP_{BV*BT-1}> to the tokenizer, resizes embeddings.
    Step 1B: Splits logits into normal (text) and ZIP (joint introspection).
    Step 1C: Masks ZIP logits to -inf in generation_logits so they're never sampled.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        bv: int = DEFAULT_BV,
        bt: int = DEFAULT_BT,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.bv = bv
        self.bt = bt
        self.num_zip_tokens = bv * bt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.original_vocab_size = len(self.tokenizer)

        # Step 1A — Reserve ZIP tokens
        zip_tokens = [f"<ZIP_{i}>" for i in range(self.num_zip_tokens)]
        self.tokenizer.add_tokens(zip_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.lm_head.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = outputs.logits  # (B, T, V + num_zip_tokens)

        # Step 1B — Split
        normal_logits = logits[..., :self.original_vocab_size]
        zip_logits = logits[..., -self.num_zip_tokens:]  # (B, T, BV*BT)

        # Step 1C — Mask ZIP tokens for generation
        generation_logits = logits.clone()
        generation_logits[..., -self.num_zip_tokens:] = float("-inf")

        return ZipRCOutput(
            token_logits=normal_logits,
            generation_logits=generation_logits,
            zip_logits=zip_logits,
        )

    def predict_zip(self, input_ids, attention_mask=None):
        """Return marginal reward and length distributions from the joint ZIP logits."""
        out = self.forward(input_ids, attention_mask)
        # Joint distribution over (reward_bin, length_bin) at last position
        joint_probs = torch.softmax(out.zip_logits[:, -1, :], dim=-1)  # (B, BV*BT)
        joint_grid = joint_probs.view(-1, self.bv, self.bt)  # (B, BV, BT)

        # Marginals
        reward_probs = joint_grid.sum(dim=-1)  # (B, BV) — sum over length bins
        length_probs = joint_grid.sum(dim=-2)  # (B, BT) — sum over reward bins

        # Expected values
        reward_vals = torch.arange(self.bv, device=joint_probs.device, dtype=torch.float)
        length_vals = torch.arange(self.bt, device=joint_probs.device, dtype=torch.float)

        expected_reward = (reward_probs * reward_vals).sum(dim=-1)
        expected_length = (length_probs * length_vals).sum(dim=-1)

        return expected_reward, expected_length


class ZipRCOutput:
    __slots__ = ("token_logits", "generation_logits", "zip_logits")

    def __init__(self, token_logits, generation_logits, zip_logits):
        self.token_logits = token_logits
        self.generation_logits = generation_logits
        self.zip_logits = zip_logits


if __name__ == "__main__":
    model = ZipRCModel("Qwen/Qwen2.5-1.5B-Instruct", bv=2, bt=5)
    tok = model.tokenizer
    inputs = tok("Hello, world!", return_tensors="pt")

    out = model(**inputs)
    print(f"Token logits shape: {out.token_logits.shape}")
    print(f"ZIP logits shape:   {out.zip_logits.shape}")  # (1, T, 10)

    reward, length = model.predict_zip(**inputs)
    print(f"Expected reward: {reward.item():.3f}")
    print(f"Expected length bin: {length.item():.1f}")

    # Verify Step 1C
    assert (out.generation_logits[..., -model.num_zip_tokens:] == float("-inf")).all()
    probs = torch.softmax(out.generation_logits[:, -1, :], dim=-1)
    assert probs[0, -model.num_zip_tokens:].sum() == 0
    print("Step 1C verified: ZIP tokens have 0 sampling probability")

    # Verify joint label helper
    assert joint_label(0, 3) == 3
    assert joint_label(1, 3) == 8
    assert tokens_left_to_bin(0) == 0
    assert tokens_left_to_bin(5) == 2
    assert tokens_left_to_bin(20) == 4
    print("Joint label helpers verified")
