"""
Train ZIP-RC Model

Trains the ZipRCModel to predict joint (reward, length) labels from prefix examples.

Loss function (from paper Section 4, Equation 3):
    L = L_aux + α_KL × KL(π || π_θ)

Where:
    - L_aux: Cross-entropy loss on ZIP logits
    - KL term: Regularization to preserve base model's generation quality
    - α_KL: Weighting factor for KL regularization (0 = no regularization)

Usage:
    python train_zip_rc.py --epochs 3 --batch_size 16 --learning_rate 5e-5 --freeze_backbone

    # With KL regularization to preserve generation quality
    python train_zip_rc.py --epochs 3 --batch_size 16 --alpha_kl 0.1

    # Full training (unfreeze backbone)
    python train_zip_rc.py --epochs 5 --batch_size 8 --learning_rate 1e-5 --no-freeze_backbone
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from zip_rc_model import ZipRCModel, DEFAULT_BV, DEFAULT_BT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PrefixDataset(Dataset):
    """Dataset for prefix examples with joint (reward, length) labels."""

    def __init__(self, jsonl_path: str):
        """
        Load prefix dataset from JSONL file.

        Each line contains:
            - input_ids: list of token IDs
            - attention_mask: list of 1s
            - joint_label: int in [0, BV*BT-1]
        """
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append({
                    "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
                    "joint_label": ex["joint_label"],
                })

        logger.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad sequences to same length within batch."""
    # Find max length in batch
    max_len = max(ex["input_ids"].size(0) for ex in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for ex in batch:
        seq_len = ex["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Left-pad input_ids with 0 (padding token)
        padded_ids = F.pad(ex["input_ids"], (pad_len, 0), value=0)
        padded_mask = F.pad(ex["attention_mask"], (pad_len, 0), value=0)

        input_ids.append(padded_ids)
        attention_masks.append(padded_mask)
        labels.append(ex["joint_label"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def compute_kl_divergence(
    model: ZipRCModel,
    base_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence KL(π_base || π_θ) over text tokens.

    This regularization term preserves the base model's generation quality
    by penalizing drift from the original token distribution.
    """
    with torch.no_grad():
        base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = base_outputs.logits[..., : model.original_vocab_size]
        base_log_probs = F.log_softmax(base_logits, dim=-1)

    # Get current model's text logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    current_logits = outputs.token_logits
    current_log_probs = F.log_softmax(current_logits, dim=-1)

    # KL(base || current) = sum(p_base * (log p_base - log p_current))
    # Use last position only (matching paper's approach)
    kl_div = F.kl_div(
        current_log_probs[:, -1, :],
        base_log_probs[:, -1, :],
        reduction="batchmean",
        log_target=True,
    )

    return kl_div


def train_epoch(
    model: ZipRCModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    alpha_kl: float = 0.0,
    base_model: Optional[nn.Module] = None,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_aux_loss = 0.0
    total_kl_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # L_aux: Cross-entropy on ZIP logits at last position
        zip_logits = outputs.zip_logits[:, -1, :]  # (B, BV*BT)
        aux_loss = criterion(zip_logits, labels)

        # Optional KL regularization
        kl_loss = torch.tensor(0.0, device=device)
        if alpha_kl > 0 and base_model is not None:
            kl_loss = compute_kl_divergence(model, base_model, input_ids, attention_mask)

        # Total loss
        loss = aux_loss + alpha_kl * kl_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_aux_loss += aux_loss.item()
        total_kl_loss += kl_loss.item() if alpha_kl > 0 else 0.0

        preds = zip_logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        acc = correct / total if total > 0 else 0.0
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "aux": f"{aux_loss.item():.4f}",
            "kl": f"{kl_loss.item():.4f}" if alpha_kl > 0 else "N/A",
            "acc": f"{acc:.2%}",
        })

    return {
        "loss": total_loss / len(dataloader),
        "aux_loss": total_aux_loss / len(dataloader),
        "kl_loss": total_kl_loss / len(dataloader) if alpha_kl > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def save_checkpoint(
    model: ZipRCModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: dict,
    checkpoint_dir: Path,
):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"

    # Save model state (only trainable parameters)
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if any(p.requires_grad for p in [model.state_dict()[k]] if isinstance(model.state_dict()[k], torch.Tensor))
    }

    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "bv": model.bv,
            "bt": model.bt,
            "original_vocab_size": model.original_vocab_size,
        },
    }, checkpoint_path)

    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Also save latest checkpoint pointer
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "bv": model.bv,
            "bt": model.bt,
            "original_vocab_size": model.original_vocab_size,
        },
    }, latest_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train ZIP-RC model on prefix dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/prefix_dataset.jsonl",
        help="Path to prefix dataset JSONL file",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze backbone, only train LM head",
    )
    parser.add_argument(
        "--no-freeze_backbone",
        action="store_false",
        dest="freeze_backbone",
        help="Train full model (unfreeze backbone)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--alpha_kl",
        type=float,
        default=0.0,
        help="Weight for KL regularization term (0 = disabled)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Validate data path
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Dataset not found: {args.data_path}\n"
            "Run the data pipeline first:\n"
            "  python generate_rollouts.py\n"
            "  python score_rollouts.py\n"
            "  python build_prefix_dataset.py"
        )

    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    dataset = PrefixDataset(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Keep simple for compatibility
    )

    # Initialize model
    logger.info(f"Loading model: {args.model_name}")
    logger.info(f"Freeze backbone: {args.freeze_backbone}")

    model = ZipRCModel(
        model_name_or_path=args.model_name,
        bv=DEFAULT_BV,
        bt=DEFAULT_BT,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Load base model for KL regularization if needed
    base_model = None
    if args.alpha_kl > 0:
        logger.info(f"Loading base model for KL regularization (alpha_kl={args.alpha_kl})")
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        base_model = base_model.to(device)
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Alpha KL: {args.alpha_kl}")
    logger.info(f"  Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"  Checkpoint steps: {args.checkpoint_steps}")
    logger.info("=" * 60)

    checkpoint_dir = Path(args.checkpoint_dir)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")

        # Train epoch with step-level checkpointing
        model.train()
        total_loss = 0.0
        total_aux_loss = 0.0
        total_kl_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # L_aux: Cross-entropy on ZIP logits at last position
            zip_logits = outputs.zip_logits[:, -1, :]
            aux_loss = criterion(zip_logits, labels)

            # Optional KL regularization
            kl_loss = torch.tensor(0.0, device=device)
            if args.alpha_kl > 0 and base_model is not None:
                kl_loss = compute_kl_divergence(model, base_model, input_ids, attention_mask)

            # Total loss
            loss = aux_loss + args.alpha_kl * kl_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_aux_loss += aux_loss.item()
            total_kl_loss += kl_loss.item() if args.alpha_kl > 0 else 0.0

            preds = zip_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            global_step += 1

            # Update progress bar
            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.2%}",
            })

            # Save checkpoint every N steps
            if global_step % args.checkpoint_steps == 0:
                metrics = {
                    "loss": total_loss / (batch_idx + 1),
                    "aux_loss": total_aux_loss / (batch_idx + 1),
                    "kl_loss": total_kl_loss / (batch_idx + 1) if args.alpha_kl > 0 else 0.0,
                    "accuracy": correct / total if total > 0 else 0.0,
                }
                save_checkpoint(model, optimizer, epoch, global_step, metrics, checkpoint_dir)

        # End of epoch metrics
        epoch_metrics = {
            "loss": total_loss / len(dataloader),
            "aux_loss": total_aux_loss / len(dataloader),
            "kl_loss": total_kl_loss / len(dataloader) if args.alpha_kl > 0 else 0.0,
            "accuracy": correct / total if total > 0 else 0.0,
        }

        logger.info(f"Epoch {epoch} complete:")
        logger.info(f"  Loss: {epoch_metrics['loss']:.4f}")
        logger.info(f"  Aux Loss: {epoch_metrics['aux_loss']:.4f}")
        if args.alpha_kl > 0:
            logger.info(f"  KL Loss: {epoch_metrics['kl_loss']:.4f}")
        logger.info(f"  Accuracy: {epoch_metrics['accuracy']:.2%}")

        # Save end-of-epoch checkpoint
        save_checkpoint(model, optimizer, epoch, global_step, epoch_metrics, checkpoint_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Final checkpoint saved to {checkpoint_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
