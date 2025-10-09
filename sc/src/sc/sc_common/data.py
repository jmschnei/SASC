# sc_common/data.py — datasets + tokenizers + loaders (shared)
import random, numpy as np
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_tokenizer(model_name:str):
    return AutoTokenizer.from_pretrained(model_name)

# ---------------- Flat (HF dataset) ----------------
class FlatHFDataset(Dataset):
    def __init__(self, texts:List[str], labels:List[int], tok, max_length:int=256):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(str(self.texts[i]), truncation=True, padding='max_length',
                       max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(int(self.labels[i]), dtype=torch.long)
        }

def build_flat_hf_dataloaders(hf_name:str, tok, batch_size:int=16, max_length:int=256, num_workers:int=2):
    ds = load_dataset(hf_name)
    feat = ds['train'].features['label']
    if isinstance(feat, ClassLabel) and hasattr(feat, 'names'):
        label_names = list(feat.names)
    else:
        n = int(max(ds['train']['label'])) + 1
        label_names = [f'Label{i}' for i in range(n)]

    def to_lists(split): return ds[split]['text'], ds[split]['label']
    tr_texts, tr_labels = to_lists('train')
    va_texts, va_labels = to_lists('validation')
    te_texts, te_labels = to_lists('test')

    tr = FlatHFDataset(tr_texts, tr_labels, tok, max_length)
    va = FlatHFDataset(va_texts, va_labels, tok, max_length)
    te = FlatHFDataset(te_texts, te_labels, tok, max_length)

    print("=== DEBUG ===")
    print("Train len:", len(tr))
    print("Valid len:", len(va))

    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_va = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_te = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_tr, dl_va, dl_te, label_names

# ---------------- Hierarchical (CSV) ----------------
class HierCSVSectionDataset(Dataset):
    """CSV columns: text, level1, level2 (level2 optional)"""
    def __init__(self, csv_path:str, tok, l1_to_idx:Dict[str,int], l2_to_idx:Dict[str,Dict[str,int]], max_len:int=256):
        import csv
        self.rows, self.tok, self.max_len = [], tok, max_len
        self.l1_to_idx, self.l2_to_idx = l1_to_idx, l2_to_idx
        with open(csv_path, newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                text = (r.get('text') or '').strip()
                l1   = (r.get('level1') or '').strip()
                l2   = (r.get('level2') or '').strip()
                if not text or (l1 not in l1_to_idx): continue
                self.rows.append((text, l1, l2))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        text, l1, l2 = self.rows[i]
        enc = self.tok(text, truncation=True, padding='max_length',
                       max_length=self.max_len, return_tensors='pt')
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'l1': torch.tensor(self.l1_to_idx[l1], dtype=torch.long),
            'l1_name': l1
        }
        if l1 in self.l2_to_idx and l2 and (l2 in self.l2_to_idx[l1]):
            item['l2'] = torch.tensor(self.l2_to_idx[l1][l2], dtype=torch.long)
            item['has_l2'] = torch.tensor(1, dtype=torch.bool)
        else:
            item['l2'] = torch.tensor(0, dtype=torch.long)
            item['has_l2'] = torch.tensor(0, dtype=torch.bool)
        return item

def build_hier_csv_dataloaders(train_csv:str, eval_csv:str, tok, l1_to_idx, l2_to_idx,
                               batch_size:int=8, max_length:int=256, num_workers:int=2):
    tr = HierCSVSectionDataset(train_csv, tok, l1_to_idx, l2_to_idx, max_length)
    va = HierCSVSectionDataset(eval_csv,   tok, l1_to_idx, l2_to_idx, max_length)
    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_va = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_tr, dl_va
