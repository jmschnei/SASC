"""
SASC using SciBERT
==================================================================
"""
# data: https://huggingface.co/datasets/nhop/academic-section-classification/viewer

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import wandb
import numpy as np
import pandas as pd
import random
import json
import argparse
from datasets import load_dataset, ClassLabel
from typing import List, Dict, Tuple
import warnings

# --------------------- Argument Parser ---------------------
def build_arg_parser():
    p = argparse.ArgumentParser()

    # MODEL_ARGS
    p.add_argument("--model_type", type=str, default="sc")
    p.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased")
    p.add_argument("--tokenizer_name", type=str, default="allenai/scibert_scivocab_uncased")
    p.add_argument("--config_overrides", type=str, default="")

    # DATA_ARGS
    p.add_argument("--dataset_name", type=str, default="nhop/academic-section-classification")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    # TRAINING_ARGS
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_strategy", type=str, default="steps", choices=["steps","epoch"])
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=4)

    # RUN_ARGS
    p.add_argument("--run_name", type=str, default="run")
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--output_dir", type=str, default="./outputs")

    return p

class ScientificSectionDataset(Dataset):
    """Dataset for scientific section classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SciBERTSectionClassifier(nn.Module):
    """Clasifier based on SciBERT for scientific section classification"""
    
    def __init__(self, n_classes: int, model_name: str = 'allenai/scibert_scivocab_uncased'):
        super(SciBERTSectionClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(
        model_name,
        use_safetensors=True,
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


class SectionClassificationSystem:
    """Whole system to classify scientific sections"""
    
    # Estandar mapping of scientific sections
    SECTION_LABELS = {
        'abstract': 0,
        'introduction': 1,
        'background': 2,
        'methods': 3,
        'results': 4,
        'discussion': 5,
        'conclusion': 6,
        # 'references': 7,
        # 'acknowledgments': 8,
        'supplementary': 9
    }
    
    def __init__(self, model_name: str = 'allenai/scibert_scivocab_uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_to_section = {v: k for k, v in self.SECTION_LABELS.items()}

    def save_artifacts(self, output_dir: str, label_names: List[str]):
        """Save model weights, tokenizer, and label list to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        # save model weights
        try:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        except Exception as e:
            print(f"[warn] torch.save failed: {e}")
        # save tokenizer
        try:
            self.tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"[warn] tokenizer.save_pretrained failed: {e}")
        # save labels
        with open(os.path.join(output_dir, "labels.txt"), "w", encoding="utf-8") as f:
            for i, name in enumerate(label_names):
                f.write(f"{i}\t{name}\n")
        print(f"✓ Saved artifacts to {output_dir}")

    def set_label_space(self, label_names):
        """
        Use the dataset’s label names to dynamically replace the label space inside the class.
        ex: ["Introduction","Background","Methodology","Experiments and Results","Conclusion"]
        """
        self.SECTION_LABELS = {name.lower(): i for i, name in enumerate(label_names)}
        self.label_to_section = {i: name for name, i in self.SECTION_LABELS.items()}

    def dataloaders_from_hf(self, dataset_name: str, batch_size: int = 16, max_length: int = 256):
        """
        Load train/val/test directly from HuggingFace, and return the three DataLoaders plus label_names.
        Compatible with both cases where the label feature is a ClassLabel or a plain integer.
        """
        ds = load_dataset(dataset_name)

        feat = ds["train"].features["label"]
        if isinstance(feat, ClassLabel) and hasattr(feat, "names"):
            label_names = list(feat.names)
        else:
            label_names = [
                "Introduction",
                "Background",
                "Methodology",
                "Experiments and Results",
                "Conclusion",
            ]

        self.set_label_space(label_names)

        def split_to_lists(split):
            texts = ds[split]["text"]
            labels = ds[split]["label"]  # already with integer: 0..n-1
            return texts, labels

        train_texts, train_labels = split_to_lists("train")
        val_texts,   val_labels   = split_to_lists("validation")
        test_texts,  test_labels  = split_to_lists("test")

        train_dataset = ScientificSectionDataset(train_texts, train_labels, self.tokenizer, max_length=max_length)
        val_dataset   = ScientificSectionDataset(val_texts,   val_labels,   self.tokenizer, max_length=max_length)
        test_dataset  = ScientificSectionDataset(test_texts,  test_labels,  self.tokenizer, max_length=max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size)
        
        print(f"✓ Labels detected: {label_names}")

        return train_loader, val_loader, test_loader, label_names

    def prepare_data(self, texts: List[str], labels: List[str]) -> Tuple:
        """Prepare data for training"""
        # Convert text labels to numbers
        label_ids = [self.SECTION_LABELS.get(label.lower(), -1) for label in labels]
        
        # Filter examples
        valid_data = [(t, l) for t, l in zip(texts, label_ids) if l != -1]
        texts, label_ids = zip(*valid_data) if valid_data else ([], [])
        
        return list(texts), list(label_ids)
    
    def create_data_loaders(self, texts: List[str], labels: List[int], 
                          test_size: float = 0.2, batch_size: int = 16) -> Tuple:
        """Create DataLoaders for training and validation"""
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = ScientificSectionDataset(X_train, y_train, self.tokenizer)
        val_dataset = ScientificSectionDataset(X_val, y_val, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 5, learning_rate: float = 2e-5, wandb_run=None) -> Dict:
        """Train the model"""
        
        # Inicialize model
        n_classes = len(self.SECTION_LABELS)
        self.model = SciBERTSectionClassifier(n_classes).to(self.device)
        
        # Configure optimizer and loss function
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Trainig
            self.model.train()
            train_loss_sum = 0.0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss_sum += loss.item()
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Store metrics
            epoch_train_loss = train_loss_sum / max(len(train_loader), 1)
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {epoch_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print("-" * 50)

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": epoch_train_loss,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                })

        return history
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def report_from_loader(self, data_loader, label_names, return_arrays=False):
        """
        Compute the classification report directly from a DataLoader (using the true integer labels).
        optionally return arrays for confusion matrix.
        """
        self.model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].cpu().numpy()
                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_trues.extend(labels.tolist() if hasattr(labels, "tolist") else list(labels))
                all_preds.extend(preds.tolist())

        report = classification_report(all_trues, all_preds, target_names=label_names, zero_division=0)
        if return_arrays:
            return report, all_trues, all_preds
        return report

    def predict(self, texts: List[str]) -> List[Dict]:
        """Predict the sections for a given list of texts"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                predictions.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'predicted_section': self.label_to_section[predicted_class],
                    'confidence': confidence,
                    'all_probabilities': {
                        self.label_to_section[i]: prob.item() 
                        for i, prob in enumerate(probabilities[0])
                    }
                })
        
        return predictions
    
    def generate_classification_report(self, texts: List[str], true_labels: List[str]) -> str:
        """Generate a detailed classification report"""
        _, true_label_ids = self.prepare_data(texts, true_labels)
        predictions = self.predict(texts)
        pred_labels = [self.SECTION_LABELS[p['predicted_section']] for p in predictions]
        
        report = classification_report(
            true_label_ids, 
            pred_labels,
            target_names=list(self.SECTION_LABELS.keys()),
            zero_division=0
        )
        
        return report


# ====================
# EXAMPLE DATA
# ====================

def generate_sample_data() -> Tuple[List[str], List[str]]:
    """Generate example data for a demo"""
    
    sample_texts = [
        # Abstract
        "This study investigates the effects of climate change on marine biodiversity. We analyzed data from 50 coastal regions over 10 years.",
        "We present a novel approach to quantum computing that reduces error rates by 40%. Our method combines topological protection with error correction.",
        
        # Introduction
        "Machine learning has revolutionized many fields in recent years. The ability to automatically learn patterns from data has enabled breakthrough applications.",
        "Cancer remains one of the leading causes of death worldwide. Understanding the molecular mechanisms underlying tumor formation is crucial.",
        
        # Methods
        "Participants were randomly assigned to control and experimental groups. Data was collected using standardized questionnaires administered at baseline and follow-up.",
        "We used Python 3.8 with TensorFlow 2.0 for all experiments. The model was trained on 4 NVIDIA V100 GPUs for 72 hours.",
        
        # Results
        "The treatment group showed significant improvement (p < 0.001) compared to control. Mean scores increased from 45.2 to 67.8 after intervention.",
        "Our algorithm achieved 95.3% accuracy on the test set, outperforming the baseline by 12 percentage points.",
        
        # Discussion
        "These findings suggest that the proposed method is effective for the target population. However, several limitations should be considered.",
        "Our results align with previous studies showing similar patterns. The implications for clinical practice are substantial.",
        
        # Conclusion
        "In conclusion, this research demonstrates the feasibility of the proposed approach. Future work should focus on scaling to larger datasets.",
        "We have shown that combining these techniques yields superior performance. This opens new avenues for research in the field."
    ]
    
    sample_labels = [
        'abstract', 'abstract',
        'introduction', 'introduction',
        'methods', 'methods',
        'results', 'results',
        'discussion', 'discussion',
        'conclusion', 'conclusion'
    ]
    
    return sample_texts, sample_labels


# ====================
# FUNCIÓN PRINCIPAL
# ====================

def main():
    """Main function to demo the system"""

    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print(f"RUN: {args.run_name}")
    print("=" * 60)

    # make sure the output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- W&B ----
    wandb_run = None
    if args.report_to.lower() == "wandb":
        try:
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT"),
                name=args.run_name,
                tags=os.environ.get("WANDB_TAGS","").split(",") if os.environ.get("WANDB_TAGS") else None,
                dir=os.environ.get("WANDB_DIR") or None,
                config=vars(args),
            )
        except Exception as e:
            print(f"[warn] W&B init failed; continuing without logging: {e}")
            wandb_run = None
    
    print("\n→ Inicializando modelo SciBERT...")
    classifier = SectionClassificationSystem(model_name=args.model_name)

    print("\n→ Cargando dataset: nhop/academic-section-classification ...")
    train_loader, val_loader, test_loader, label_names = classifier.dataloaders_from_hf(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    print(f"✓ DataLoaders preparados (train/val/test)")

    print("\n→ Entrenando modelo ...")
    print("-" * 50)
    if args.do_train:
        print("starting traning")
        history = classifier.train(train_loader, val_loader, epochs=args.epochs, learning_rate=args.learning_rate, wandb_run=wandb_run)
    print("finishing traning")

    if args.do_eval:
        print("do eval")
        val_loss, val_acc = classifier.evaluate(val_loader)
        print(f"\n[VAL] loss={val_loss:.4f}, acc={val_acc:.4f}")
        if wandb_run is not None:
            wandb_run.log({"final/val_loss": val_loss, "final/val_acc": val_acc})

        print("do test")
        test_loss, test_acc = classifier.evaluate(test_loader)
        print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")
        if wandb_run is not None:
            wandb_run.log({"final/test_loss": test_loss, "final/test_acc": test_acc})

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT (TEST)")
        print("=" * 60)
        report, y_true, y_pred = classifier.report_from_loader(test_loader, label_names, return_arrays=True)
        print(report)

        # Log summary metrics + confusion matrix
        if wandb_run is not None:
            try:
                from sklearn.metrics import classification_report as cr
                rep = cr(y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True)
                wandb_run.log({
                    "report/accuracy": rep.get("accuracy"),
                    "report/macro_f1": rep["macro avg"]["f1-score"],
                    "report/weighted_f1": rep["weighted avg"]["f1-score"],
                })
                import wandb as _wandb
                cm = _wandb.plot.confusion_matrix(
                    probs=None, y_true=y_true, preds=y_pred, class_names=label_names
                )
                wandb_run.log({"confusion_matrix": cm})
            except Exception as e:
                print(f"[warn] logging report/cm failed: {e}")

    classifier.save_artifacts(args.output_dir, label_names)

    if wandb_run is not None:
        wandb_run.finish()
    
    print("\n" + "=" * 60)
    print("Ready to use")
    print("=" * 60)

# Execute main script
if __name__ == "__main__":
    main()