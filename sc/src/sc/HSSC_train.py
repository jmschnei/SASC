#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for Hierarchical SciBERT classifier (HSSC),
compatible with generic sc_scibert_train.sh-style arguments.

Main behavior:
  - Read a CSV file containing text, level-1 labels, and level-2 labels.
  - Automatically build the label hierarchy from level1 / level2 columns.
  - Train a two-level classifier using HSSC.HierarchicalSciBERT.

Default column names are adapted to scilake_cancer_level2_hssc.csv:
  - text column: section_content
  - level-1 label: hssc_level1
  - level-2 label: hssc_level2
"""

import os
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split

try:
    import wandb
except ImportError:
    wandb = None

# Import the hierarchical model from HSSC.py in the same directory
from HSSC import HierarchicalSciBERT  # type: ignore


# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_config_overrides(config_str: Optional[str]) -> Dict:
    """
    Parse 'key=value,key2=value2' style config_overrides into a dict.

    IMPORTANT:
      - At the moment we only parse and print these overrides.
      - They are NOT applied to HierarchicalSciBERT unless you modify HSSC.py.
    """
    if not config_str:
        return {}

    overrides = {}
    for kv in config_str.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Simple type conversion
        if value.lower() in ["true", "false"]:
            value_conv = value.lower() == "true"
        else:
            try:
                if "." in value:
                    value_conv = float(value)
                else:
                    value_conv = int(value)
            except ValueError:
                value_conv = value
        overrides[key] = value_conv
    return overrides


# =========================
# Dataset
# =========================

class HierarchicalDataset(Dataset):
    """
    Dataset for hierarchical (level1 + level2) classification.

    The level-2 label is encoded as a local index relative to its parent level-1 label,
    which matches the output structure of HierarchicalSciBERT.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str,
        level1_col: str,
        level2_col: str,
        level1_to_idx: Dict[str, int],
        level2_to_idx: Dict[str, Dict[str, int]],
        max_length: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.level1_col = level1_col
        self.level2_col = level2_col
        self.level1_to_idx = level1_to_idx
        self.level2_to_idx = level2_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        l1 = str(row[self.level1_col])
        l2 = str(row[self.level2_col])

        input_enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        level1_id = self.level1_to_idx[l1]
        level2_id = self.level2_to_idx[l1][l2]

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "level1": torch.tensor(level1_id, dtype=torch.long),
            "level2": torch.tensor(level2_id, dtype=torch.long),
            "level1_label_str": l1,
            "level2_label_str": l2,
        }


# =========================
# Argparse
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train hierarchical SciBERT classifier (HSSC)"
    )

    # --------- Core arguments (based on usage) ---------
    parser.add_argument(
        "--train_csv", type=str, required=True, help="Path to training CSV file."
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="section_content",
        help="Name of the text column in the CSV.",
    )
    parser.add_argument(
        "--level1_col",
        type=str,
        default="hssc_level1",
        help="Name of the level-1 label column in the CSV.",
    )
    parser.add_argument(
        "--level2_col",
        type=str,
        default="hssc_level2",
        help="Name of the level-2 label column in the CSV.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use as validation set.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="Base transformer model name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--alpha_level2",
        type=float,
        default=0.5,
        help="Weight of the level-2 loss in the total loss.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory where checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="hssc_run",
        help="Run name (for logging / wandb).",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="If set, log metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (fallback to env WANDB_PROJECT).",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated WandB tags (fallback to env WANDB_TAGS).",
    )
    parser.add_argument(
        "--compare_enhanced",
        action="store_true",
        help=(
            "(Placeholder) Compare with an enhanced classifier. "
            "Currently not implemented; kept only for CLI compatibility."
        ),
    )

    # --------- Extra HF-style arguments for compatibility with generic .sh ---------
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name/path. If None, uses model_name.",
    )
    parser.add_argument(
        "--config_overrides",
        type=str,
        default=None,
        help=(
            "Config overrides in 'key=value,key2=value2' format "
            "(currently parsed but not applied to HSSC)."
        ),
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="HF-style flag. Training is always run in this script; accepted for compatibility.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="HF-style flag. Evaluation on the validation set is always performed.",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default=None,
        help="HF-style eval strategy (ignored; we evaluate once per epoch).",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="HF-style eval steps (ignored).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="HF-style save steps (ignored; we save by best val loss and last epoch).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed precision via torch.cuda.amp.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default=None,
        help=(
            "HF-style reporting targets (e.g. ['wandb']). "
            "If the list contains 'wandb', we enable wandb logging."
        ),
    )

    args = parser.parse_args()
    return args


# =========================
# Training / Evaluation
# =========================

def build_hierarchy(
    df: pd.DataFrame, level1_col: str, level2_col: str
) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Build the hierarchical label structure from the data:

      Returns:
        - hierarchy_config:
            { level1_label: [level2_label1, level2_label2, ...] }
        - level1_to_idx:
            { level1_label: level1_index }
        - level2_to_idx:
            { level1_label: { level2_label: local_level2_index } }

      The local_level2_index is relative to each level1, matching how
      HierarchicalSciBERT organizes its outputs.
    """
    hierarchy_config: Dict[str, List[str]] = {}
    level1_to_idx: Dict[str, int] = {}
    level2_to_idx: Dict[str, Dict[str, int]] = {}

    level1_values = sorted(df[level1_col].astype(str).unique().tolist())
    for i, l1 in enumerate(level1_values):
        level1_to_idx[l1] = i
        subset = df[df[level1_col].astype(str) == l1]
        level2_values = sorted(subset[level2_col].astype(str).unique().tolist())
        hierarchy_config[l1] = level2_values
        level2_to_idx[l1] = {l2: j for j, l2 in enumerate(level2_values)}

    return hierarchy_config, level1_to_idx, level2_to_idx


def train_one_epoch(
    model: HierarchicalSciBERT,
    dataloader: DataLoader,
    device: torch.device,
    level1_to_idx: Dict[str, int],
    level2_to_idx: Dict[str, Dict[str, int]],
    alpha_level2: float,
    fp16: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float, float]:
    """
    Train for a single epoch.

    Returns:
      (avg_total_loss, level1_acc, level2_acc_on_samples_with_l2)
    """
    model.train()
    criterion_l1 = nn.CrossEntropyLoss()
    criterion_l2 = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    total_loss = 0.0
    total_correct_l1 = 0
    total_count = 0
    total_correct_l2 = 0
    total_count_l2 = 0

    idx_to_level1 = {v: k for k, v in level1_to_idx.items()}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        level1_targets = batch["level1"].to(device)
        level2_targets = batch["level2"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=fp16):
            level1_logits, level2_logits = model(input_ids, attention_mask)

            # Level-1 loss
            loss_l1 = criterion_l1(level1_logits, level1_targets)

            # Level-2 loss (grouped by parent level1)
            loss_l2_sum = 0.0
            l2_count = 0

            for l1_idx, l1_label in idx_to_level1.items():
                if l1_label not in level2_logits:
                    continue

                mask = (level1_targets == l1_idx)
                if mask.sum().item() == 0:
                    continue

                logits_l2_parent = level2_logits[l1_label][mask]
                targets_l2_parent = level2_targets[mask]
                loss_l2 = criterion_l2(logits_l2_parent, targets_l2_parent)

                loss_l2_sum = loss_l2_sum + loss_l2 * mask.sum().item()
                l2_count += mask.sum().item()

                # Level-2 accuracy
                preds_l2 = logits_l2_parent.argmax(dim=1)
                total_correct_l2 += (preds_l2 == targets_l2_parent).sum().item()

            if l2_count > 0:
                loss_l2_avg = loss_l2_sum / l2_count
                loss = (1.0 - alpha_level2) * loss_l1 + alpha_level2 * loss_l2_avg
            else:
                # If for some reason no valid level2 targets exist, fall back to level1 loss only
                loss = loss_l1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * input_ids.size(0)
        # Level-1 accuracy
        preds_l1 = level1_logits.argmax(dim=1)
        total_correct_l1 += (preds_l1 == level1_targets).sum().item()
        total_count += input_ids.size(0)
        total_count_l2 += l2_count

    avg_loss = total_loss / max(1, total_count)
    acc_l1 = total_correct_l1 / max(1, total_count)
    acc_l2 = total_correct_l2 / max(1, total_count_l2)

    return avg_loss, acc_l1, acc_l2


def evaluate(
    model: HierarchicalSciBERT,
    dataloader: DataLoader,
    device: torch.device,
    level1_to_idx: Dict[str, int],
    level2_to_idx: Dict[str, Dict[str, int]],
    alpha_level2: float,
    fp16: bool = False,
) -> Tuple[float, float, float]:
    """
    Evaluate on the validation set.

    Returns:
      (avg_total_loss, level1_acc, level2_acc_on_samples_with_l2)
    """
    model.eval()
    criterion_l1 = nn.CrossEntropyLoss()
    criterion_l2 = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct_l1 = 0
    total_count = 0
    total_correct_l2 = 0
    total_count_l2 = 0

    idx_to_level1 = {v: k for k, v in level1_to_idx.items()}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            level1_targets = batch["level1"].to(device)
            level2_targets = batch["level2"].to(device)

            with torch.cuda.amp.autocast(enabled=fp16):
                level1_logits, level2_logits = model(input_ids, attention_mask)

                loss_l1 = criterion_l1(level1_logits, level1_targets)

                loss_l2_sum = 0.0
                l2_count = 0

                for l1_idx, l1_label in idx_to_level1.items():
                    if l1_label not in level2_logits:
                        continue

                    mask = (level1_targets == l1_idx)
                    if mask.sum().item() == 0:
                        continue

                    logits_l2_parent = level2_logits[l1_label][mask]
                    targets_l2_parent = level2_targets[mask]
                    loss_l2 = criterion_l2(logits_l2_parent, targets_l2_parent)

                    loss_l2_sum = loss_l2_sum + loss_l2 * mask.sum().item()
                    l2_count += mask.sum().item()

                    # Level-2 accuracy
                    preds_l2 = logits_l2_parent.argmax(dim=1)
                    total_correct_l2 += (preds_l2 == targets_l2_parent).sum().item()

                if l2_count > 0:
                    loss_l2_avg = loss_l2_sum / l2_count
                    loss = (1.0 - alpha_level2) * loss_l1 + alpha_level2 * loss_l2_avg
                else:
                    loss = loss_l1

            total_loss += loss.item() * input_ids.size(0)
            preds_l1 = level1_logits.argmax(dim=1)
            total_correct_l1 += (preds_l1 == level1_targets).sum().item()
            total_count += input_ids.size(0)
            total_count_l2 += l2_count

    avg_loss = total_loss / max(1, total_count)
    acc_l1 = total_correct_l1 / max(1, total_count)
    acc_l2 = total_correct_l2 / max(1, total_count_l2)

    return avg_loss, acc_l1, acc_l2


# =========================
# Main
# =========================

def main():
    args = parse_args()

    # Seed
    set_seed(args.seed)

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # config_overrides are only parsed and printed; not applied to the model
    overrides = parse_config_overrides(args.config_overrides)
    if overrides:
        print(f"[HSSC_train] Parsed config_overrides (currently NOT applied to HSSC): {overrides}")

    # Load data
    print(f"[HSSC_train] Loading data from {args.train_csv}")
    df = pd.read_csv(args.train_csv)

    if args.text_col not in df.columns:
        raise ValueError(f"text_col '{args.text_col}' not in CSV columns: {df.columns}")
    if args.level1_col not in df.columns:
        raise ValueError(f"level1_col '{args.level1_col}' not in CSV columns: {df.columns}")
    if args.level2_col not in df.columns:
        raise ValueError(f"level2_col '{args.level2_col}' not in CSV columns: {df.columns}")

    # --- NEW: filter out rows without level2 label for hierarchical training ---
    total_rows = len(df)
    no_level2_mask = df[args.level2_col].isna()
    num_no_level2 = int(no_level2_mask.sum())

    # We keep only rows that have a non-missing level2 label
    df = df[~no_level2_mask].reset_index(drop=True)
    kept_rows = len(df)

    print(
        f"[HSSC_train] Filtered rows without level2: "
        f"removed {num_no_level2}/{total_rows}, kept {kept_rows} samples with non-empty level2."
    )
    # ---------------------------------------------------------------------------

    # Build hierarchy & label mappings
    hierarchy_config, level1_to_idx, level2_to_idx = build_hierarchy(
        df, args.level1_col, args.level2_col
    )
    print(f"[HSSC_train] Detected {len(level1_to_idx)} level1 classes.")
    for l1, children in hierarchy_config.items():
        print(f"  - {l1}: {len(children)} level2 classes")

    # Train/validation split (stratified by level1)
    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[args.level1_col].astype(str),
    )

    print(
        f"[HSSC_train] Dataset split: "
        f"train={len(train_df)} samples, val={len(val_df)} samples "
        f"(test_size={args.test_size})."
    )
    
    # Tokenizer
    tokenizer_name = args.tokenizer_name or args.model_name
    print(f"[HSSC_train] Loading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Dataset & dataloader
    train_dataset = HierarchicalDataset(
        train_df,
        tokenizer,
        args.text_col,
        args.level1_col,
        args.level2_col,
        level1_to_idx,
        level2_to_idx,
        max_length=args.max_length,
    )
    val_dataset = HierarchicalDataset(
        val_df,
        tokenizer,
        args.text_col,
        args.level1_col,
        args.level2_col,
        level1_to_idx,
        level2_to_idx,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HSSC_train] Using device: {device}")

    # Model
    print(f"[HSSC_train] Initializing HierarchicalSciBERT with model_name={args.model_name}")
    model = HierarchicalSciBERT(hierarchy_config, model_name=args.model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # wandb switch logic:
    #   1) --use_wandb explicitly set, OR
    #   2) 'wandb' appears in --report_to
    use_wandb = args.use_wandb or (args.report_to and "wandb" in args.report_to)
    if use_wandb and wandb is None:
        print("[HSSC_train] wandb is not installed, disabling wandb logging.")
        use_wandb = False

    if use_wandb:
        project = args.wandb_project or os.environ.get("WANDB_PROJECT", "hssc")
        tags_str = args.wandb_tags or os.environ.get("WANDB_TAGS", "")
        tags = [t for t in tags_str.split(",") if t] if tags_str else None

        print(
            "[HSSC_train] Initializing wandb: "
            f"project={project}, run_name={args.run_name}, tags={tags}"
        )
        wandb.init(
            project=project,
            name=args.run_name,
            tags=tags,
            config={
                "model_name": args.model_name,
                "tokenizer_name": tokenizer_name,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "alpha_level2": args.alpha_level2,
            },
        )

    if args.compare_enhanced:
        print(
            "[HSSC_train] NOTE: --compare_enhanced flag is accepted but not implemented "
            "(EnhancedHierarchicalClassifier is designed for inference-time comparison only)."
        )

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

        train_loss, train_acc_l1, train_acc_l2 = train_one_epoch(
            model,
            train_loader,
            device,
            level1_to_idx,
            level2_to_idx,
            alpha_level2=args.alpha_level2,
            fp16=args.fp16,
            optimizer=optimizer,
        )

        val_loss, val_acc_l1, val_acc_l2 = evaluate(
            model,
            val_loader,
            device,
            level1_to_idx,
            level2_to_idx,
            alpha_level2=args.alpha_level2,
            fp16=args.fp16,
        )

        print(
            f"[Train] loss={train_loss:.4f}, "
            f"acc_l1={train_acc_l1:.4f}, acc_l2={train_acc_l2:.4f}"
        )
        print(
            f"[Val]   loss={val_loss:.4f}, "
            f"acc_l1={val_acc_l1:.4f}, acc_l2={val_acc_l2:.4f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/acc_level1": train_acc_l1,
                    "train/acc_level2": train_acc_l2,
                    "val/loss": val_loss,
                    "val/acc_level1": val_acc_l1,
                    "val/acc_level2": val_acc_l2,
                    "epoch": epoch,
                }
            )

        # Simple "best checkpoint" saving based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            print(f"[HSSC_train] Saving best model to {ckpt_path}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hierarchy_config": hierarchy_config,
                    "level1_to_idx": level1_to_idx,
                    "level2_to_idx": level2_to_idx,
                    "args": vars(args),
                },
                ckpt_path,
            )

    # Save a final "last epoch" checkpoint
    last_ckpt_path = os.path.join(args.output_dir, "last_model.pt")
    print(f"[HSSC_train] Saving last model to {last_ckpt_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hierarchy_config": hierarchy_config,
            "level1_to_idx": level1_to_idx,
            "level2_to_idx": level2_to_idx,
            "args": vars(args),
        },
        last_ckpt_path,
    )

    if use_wandb:
        wandb.finish()

    print("[HSSC_train] Training finished.")


if __name__ == "__main__":
    main()
