#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for a 1-level SciBERT classifier, compatible with sc_scibert_train.sh
and current --train_csv usage.

It supports two data sources:
  1) Local CSV via:
       --train_csv /path/to.csv
     (default text column: section_content; label column: hssc_level1)
  2) HuggingFace Hub dataset via:
       --dataset_name nhop/academic-section-classification

It accepts HF-style training arguments (do_train, do_eval, fp16, eval_steps,
save_steps, etc.), but internally uses a custom training loop (no Trainer).
Extra arguments are parsed and ignored safely.

"""

import os
import argparse
import random
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

# Reuse existing model definition
from SciBERT_Classifier import SciBERTSectionClassifier  # type: ignore


print(">>> [SciBERT_Classifier_train] Script loaded (version with --train_csv, label_map & parse_known_args).")


# =====================
# Utils
# =====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_config_overrides(config_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse a \"key=value,key2=value2\" string into a dict.
    For now we only parse and print it; it is NOT applied to SciBERTSectionClassifier
    unless you modify SciBERT_Classifier.py accordingly.
    """
    if not config_str:
        return {}
    overrides = {}
    for kv in config_str.split(","):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Simple type conversion
        if v.lower() in ("true", "false"):
            v_conv = v.lower() == "true"
        else:
            try:
                if "." in v:
                    v_conv = float(v)
                else:
                    v_conv = int(v)
            except ValueError:
                v_conv = v
        overrides[k] = v_conv
    return overrides


# =====================
# Dataset
# =====================

class HFDatasetWrapper(Dataset):
    """
    Wrap a HuggingFace Dataset as a PyTorch Dataset and tokenize with SciBERT's tokenizer.

    Supported label formats:
      - label is already int (HF ClassLabel)
      - label is a string (e.g. 'abstract') that we map to int via label_map

    During __init__, we filter out:
      - samples where label is None
      - samples whose label is not in label_map (if label_map is provided)
    """

    def __init__(
        self,
        hf_split,
        tokenizer,
        text_column: str,
        label_column: str,
        max_length: int = 256,
        label_map: Optional[Dict[Any, int]] = None,
        split_name: str = "dataset",
    ):
        self.ds = hf_split
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.label_map = label_map
        self.split_name = split_name

        # Pre-filter invalid labels and keep indices only for valid samples
        self.valid_indices = []
        for i, ex in enumerate(self.ds):
            v = ex[self.label_column]
            if v is None:
                continue
            if self.label_map is not None:
                key = str(v)
                if key not in self.label_map:
                    # Example: label appears in val but not in train; skip it
                    continue
            self.valid_indices.append(i)

        print(
        f"[HFDatasetWrapper:{self.split_name}] Kept {len(self.valid_indices)}/{len(self.ds)} "
        f"samples for text='{self.text_column}', label='{self.label_column}'"
    )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.ds[real_idx]
        text = str(item[self.text_column])
        raw_label = item[self.label_column]

        if self.label_map is not None:
            key = str(raw_label)
            # This should not raise KeyError because __init__ has already filtered
            label = self.label_map[key]
        else:
            # Assume dataset already stores integer labels (HF ClassLabel)
            label = int(raw_label)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text,
        }


# =====================
# Argparse
# =====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SciBERT section classifier (1-level)."
    )

    # ---- MODEL ARGS ----
    parser.add_argument(
        "--model_type",
        type=str,
        default="sc_flat",
        help="Model type string (for compatibility, not actually used).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="Base transformer model name.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name/path; defaults to model_name if not set.",
    )
    parser.add_argument(
        "--config_overrides",
        type=str,
        default=None,
        help="Config overrides in 'key=value,key2=value2' format (parsed but not used).",
    )

    # ---- DATA ARGS ----
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nhop/academic-section-classification",
        help=(
            "HuggingFace dataset name. "
            "Defaults to 'nhop/academic-section-classification'."
        ),
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help=(
            "Optional local dataset files (e.g., 'train.csv,validation.csv'). "
            "If provided, this takes precedence over dataset_name."
        ),
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help=(
            "Path to a local training CSV file "
            "(e.g., scilake_cancer_level2_hssc.csv). "
            "If provided, this takes precedence over dataset_name and data_files."
        ),
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="section_content",  # default for hssc preprocessed CSV
        help="Name of the text column in dataset/CSV.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="hssc_level1",  # default: use level1 as the 1-level classification target
        help="Name of the label column in dataset/CSV.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
    )

    # ---- TRAINING ARGS (HF-style) ----
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Compatibility flag; training is always run if this script is called.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Compatibility flag; we always perform validation in this script.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        help="Eval strategy (ignored, but accepted for compatibility).",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Eval steps (ignored).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save steps (ignored; we save by epoch).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training (torch.cuda.amp).",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader.",
    )

    # ---- LOGGING / OUTPUT ----
    parser.add_argument(
        "--run_name",
        type=str,
        default="scibert_run",
        help="Run name (for logging / wandb).",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default=None,
        help="Reporting backends, e.g. ['wandb'] (if contains 'wandb', we log to wandb).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints.",
    )

    # Important: use parse_known_args so unknown args are safely ignored
    # (e.g., if the .sh script passes extra flags in the future).
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f">>> [SciBERT_Classifier_train] Ignoring unknown arguments: {unknown}")

    return args


# =====================
# Train / Eval
# =====================

def evaluate(model, dataloader, device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += input_ids.size(0)

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc


def main():
    args = parse_args()

    # Seed
    set_seed(args.seed)

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # config_overrides: only parse and print
    overrides = parse_config_overrides(args.config_overrides)
    if overrides:
        print(
            "[SciBERT_Classifier_train] Parsed config_overrides "
            "(NOT applied to SciBERTSectionClassifier): "
            f"{overrides}"
        )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SciBERT_Classifier_train] Using device: {device}")

    # Tokenizer
    tokenizer_name = args.tokenizer_name or args.model_name
    print(f"[SciBERT_Classifier_train] Loading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # =====================
    # Data loading logic:
    #   1) If --train_csv is provided, use the local CSV.
    #   2) Else, if --data_files is provided, load local HF CSV files.
    #   3) Else, load from HF Hub via --dataset_name.
    # =====================
    if args.train_csv:
        print(f"[SciBERT_Classifier_train] Loading CSV from --train_csv: {args.train_csv}")
        data_files = {"train": args.train_csv}
        raw_datasets = load_dataset("csv", data_files=data_files)
    elif args.data_files:
        print(f"[SciBERT_Classifier_train] Loading local data_files from {args.data_files}")
        # Support comma-separated multiple files; here we simply use the first as train,
        # the second (if present) as validation.
        paths = [p.strip() for p in args.data_files.split(",") if p.strip()]
        if len(paths) == 1:
            data_files = {"train": paths[0]}
        elif len(paths) >= 2:
            data_files = {"train": paths[0], "validation": paths[1]}
        else:
            raise ValueError(f"Invalid --data_files: {args.data_files}")
        raw_datasets = load_dataset("csv", data_files=data_files)
    else:
        ds_name = args.dataset_name or "nhop/academic-section-classification"
        print(f"[SciBERT_Classifier_train] Loading dataset from HF: {ds_name}")
        raw_datasets = load_dataset(ds_name)

    # Try to obtain train / validation splits
    if "train" not in raw_datasets:
        raise ValueError(f"No 'train' split found in dataset: {raw_datasets}")
    train_split = raw_datasets["train"]

    if "validation" in raw_datasets:
        val_split = raw_datasets["validation"]
    elif "validation_matched" in raw_datasets:
        val_split = raw_datasets["validation_matched"]
    elif "test" in raw_datasets:
        # If there is no dedicated validation split, use test as validation
        val_split = raw_datasets["test"]
    else:
        # If there is truly no validation/test, create a validation split from train
        print(
            "[SciBERT_Classifier_train] No validation/test split found; "
            "creating validation split from train (80/20)."
        )
        tmp = train_split.train_test_split(test_size=0.2, seed=args.seed)
        train_split = tmp["train"]
        val_split = tmp["test"]

    print(
        f"[SciBERT_Classifier_train] Using text_column='{args.text_column}', "
        f"label_column='{args.label_column}'"
    )

    # =====================
    # Build label_map & determine num_labels
    # =====================
    label_feature = train_split.features.get(args.label_column, None)
    label_map: Optional[Dict[Any, int]] = None

    if label_feature is not None and hasattr(label_feature, "names"):
        # HF ClassLabel: dataset already uses integer labels
        num_labels = len(label_feature.names)
        label_map = None
        print(f"[SciBERT_Classifier_train] Detected HF ClassLabel with {num_labels} classes.")
    else:
        # CSV scenario: label column contains strings (e.g. 'abstract')
        # Steps:
        #   - collect all non-None labels
        #   - deduplicate & sort
        #   - build label_map: {label_str -> id}
        labels_raw = []
        for x in train_split:
            v = x[args.label_column]
            if v is None:
                continue
            labels_raw.append(str(v))
        class_names = sorted(set(labels_raw))
        label_map = {name: i for i, name in enumerate(class_names)}
        num_labels = len(class_names)

        print(f"[SciBERT_Classifier_train] Built label_map with {num_labels} classes:")
        for name, idx in label_map.items():
            print(f"  - {name!r} -> {idx}")

    # Wrap HF Dataset as PyTorch Dataset with label_map
    train_ds = HFDatasetWrapper(
        train_split,
        tokenizer,
        text_column=args.text_column,
        label_column=args.label_column,
        max_length=args.max_length,
        label_map=label_map,
        split_name="TRAIN",
    )
    val_ds = HFDatasetWrapper(
        val_split,
        tokenizer,
        text_column=args.text_column,
        label_column=args.label_column,
        max_length=args.max_length,
        label_map=label_map,
        split_name="VAL",
    )

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    # Model: use SciBERTSectionClassifier
    print(f"[SciBERT_Classifier_train] Number of labels: {num_labels}")
    print(
        "[SciBERT_Classifier_train] Initializing SciBERTSectionClassifier "
        f"with model_name={args.model_name}"
    )
    model = SciBERTSectionClassifier(n_classes=num_labels, model_name=args.model_name).to(device)

    # =====================
    # Freeze SciBERT encoder (NO fine-tuning)
    # =====================
    for name, param in model.bert.named_parameters():
        param.requires_grad = False

    print("[SciBERT_Classifier_train] SciBERT encoder frozen. Training classifier head only.")

    # sanity check if it is linear probe
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())

    print(
        f"[Debug] Trainable params: {num_trainable:,} / {num_total:,} "
        f"({100.0 * num_trainable / num_total:.4f}%)"
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate) # only optimize the classifier head 's parameters (explicitly optimize only parameters with requires_grad=True.)
    criterion = nn.CrossEntropyLoss()

    # FP16
    use_fp16 = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # wandb logging
    use_wandb = args.report_to is not None and "wandb" in args.report_to
    if use_wandb and wandb is None:
        print("[SciBERT_Classifier_train] wandb is not installed; disabling wandb logging.")
        use_wandb = False

    if use_wandb:
        project = os.environ.get("WANDB_PROJECT", "sc-SciBERT")
        tags = os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else None
        print(
            "[SciBERT_Classifier_train] Initializing wandb: "
            f"project={project}, run_name={args.run_name}, tags={tags}"
        )
        wandb.init(
            project=project,
            name=args.run_name,
            tags=tags,
            config={
                "model_name": args.model_name,
                "tokenizer_name": tokenizer_name,
                "dataset_name": args.dataset_name,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "text_column": args.text_column,
                "label_column": args.label_column,
            },
        )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * input_ids.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += input_ids.size(0)

        train_loss = total_loss / max(1, total_count)
        train_acc = total_correct / max(1, total_count)

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"[Train] loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"[Val]   loss={val_loss:.4f}, acc={val_acc:.4f}")

        if use_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "epoch": epoch,
                }
            )

        # Save best model according to validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, "best_model.pt")
            print(f"[SciBERT_Classifier_train] Saving best model to {best_path}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_labels": num_labels,
                    "label_map": label_map,
                    "args": vars(args),
                },
                best_path,
            )

    # Save last epoch model
    last_path = os.path.join(args.output_dir, "last_model.pt")
    print(f"[SciBERT_Classifier_train] Saving last model to {last_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_labels": num_labels,
            "label_map": label_map,
            "args": vars(args),
        },
        last_path,
    )

    if use_wandb:
        wandb.finish()

    print("[SciBERT_Classifier_train] Training finished.")


if __name__ == "__main__":
    main()
