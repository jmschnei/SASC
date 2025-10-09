"""
Hierarchical Scientific Section Classifier
==========================================

Hierarchical classification system that leverages the natural structure
of scientific articles to improve accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import re

# [ADDED] external training/data utilities
import wandb
import os  # ADDED
import sys  # ADDED
from torch.utils.data import DataLoader  # ADDED
from transformers import get_linear_schedule_with_warmup  # ADDED
from torch.optim import AdamW
try:  # ADDED
    from datasets import load_dataset  # ADDED
except Exception as _e:  # ADDED
    load_dataset = None  # ADDED

@dataclass
class HierarchicalLabel:
    """Structure for hierarchical labels"""
    level1: str  # Main section
    level2: Optional[str] = None  # Subsection
    level3: Optional[str] = None  # Sub-subsection
    
    def to_path(self) -> str:
        """Converts the label into a hierarchical path"""
        path = [self.level1]
        if self.level2:
            path.append(self.level2)
        if self.level3:
            path.append(self.level3)
        return " > ".join(path)


class HierarchicalSciBERT(nn.Module):
    """SciBERT model with hierarchical classifiers"""
    
    def __init__(self, hierarchy_config: Dict, model_name: str = 'allenai/scibert_scivocab_uncased'):
        super(HierarchicalSciBERT, self).__init__()
        
        # Shared encoder
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size = self.bert.config.hidden_size
        
        # Hierarchy configuration
        self.hierarchy = hierarchy_config
        
        # Level classifiers
        self.level1_classifier = nn.Linear(hidden_size, len(hierarchy_config))
        self.dropout = nn.Dropout(0.3)
        
        # Level 2 classifiers (one for each level 1 category)
        self.level2_classifiers = nn.ModuleDict()
        for parent, children in hierarchy_config.items():
            if children:
                self.level2_classifiers[parent] = nn.Linear(hidden_size, len(children))
        
        # Attention to combine information between levels
        self.cross_level_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        sequence_output = outputs.last_hidden_state
        
        # Level 1 Classification
        level1_hidden = self.dropout(pooled_output)
        level1_logits = self.level1_classifier(level1_hidden)
        
        # Prepare level 2 outputs
        level2_logits = {}
        
        # For each possible level 1 class, calculate level 2 probabilities
        for parent_class in self.hierarchy.keys():
            if parent_class in self.level2_classifiers:
                # Apply cross attention to condition on level 1 prediction
                attended_output, _ = self.cross_level_attention(
                    pooled_output.unsqueeze(0),
                    sequence_output.transpose(0, 1),
                    sequence_output.transpose(0, 1)
                )
                attended_output = attended_output.squeeze(0)
                
                # Classify level 2
                level2_hidden = self.dropout(attended_output)
                level2_logits[parent_class] = self.level2_classifiers[parent_class](level2_hidden)
        
        return level1_logits, level2_logits


class HierarchicalSectionClassifier:
    """Complete hierarchical classification system"""
    
    # Scientific section hierarchy definition
    SECTION_HIERARCHY = {
        'abstract': ['objective', 'methods_summary', 'results_summary', 'conclusions_summary'],
        'introduction': ['background', 'problem_statement', 'objectives', 'contributions', 'outline'],
        'related_work': ['literature_review', 'theoretical_framework', 'gaps_identified'],
        'methods': ['study_design', 'participants', 'data_collection', 'instruments', 
                   'procedures', 'statistical_analysis', 'ethical_considerations'],
        'results': ['descriptive_statistics', 'main_findings', 'secondary_findings', 
                   'tables_figures', 'statistical_tests'],
        'discussion': ['interpretation', 'comparison_literature', 'implications', 
                      'limitations', 'future_work'],
        'conclusion': ['summary', 'key_findings', 'contributions', 'final_remarks'],
        'references': [],
        'appendix': ['supplementary_data', 'additional_analyses', 'technical_details'],
        'acknowledgments': []
    }
    
    def __init__(self, model_name: str = 'allenai/scibert_scivocab_uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = HierarchicalSciBERT(self.SECTION_HIERARCHY).to(self.device)
        
        # Index mappings
        self.level1_to_idx = {section: i for i, section in enumerate(self.SECTION_HIERARCHY.keys())}
        self.idx_to_level1 = {i: section for section, i in self.level1_to_idx.items()}
        
        self.level2_to_idx = {}
        self.idx_to_level2 = {}
        for parent, children in self.SECTION_HIERARCHY.items():
            if children:
                self.level2_to_idx[parent] = {child: i for i, child in enumerate(children)}
                self.idx_to_level2[parent] = {i: child for child, i in self.level2_to_idx[parent].items()}
    
    def predict_hierarchical(self, text: str) -> Dict:
        """Performs hierarchical prediction for a text"""
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            level1_logits, level2_logits = self.model(input_ids, attention_mask)
            
            # Level 1 Prediction
            level1_probs = F.softmax(level1_logits, dim=1)
            level1_pred = torch.argmax(level1_logits, dim=1).item()
            level1_section = self.idx_to_level1[level1_pred]
            level1_confidence = level1_probs[0][level1_pred].item()
            
            # Level 2 Prediction (if applicable)
            level2_section = None
            level2_confidence = None
            
            if level1_section in level2_logits and self.SECTION_HIERARCHY[level1_section]:
                level2_logits_section = level2_logits[level1_section]
                level2_probs = F.softmax(level2_logits_section, dim=1)
                level2_pred = torch.argmax(level2_logits_section, dim=1).item()
                level2_section = self.idx_to_level2[level1_section][level2_pred]
                level2_confidence = level2_probs[0][level2_pred].item()
        
        result = {
            'text_preview': text[:150] + '...' if len(text) > 150 else text,
            'level1': {
                'section': level1_section,
                'confidence': level1_confidence,
                'all_probabilities': {
                    self.idx_to_level1[i]: prob.item() 
                    for i, prob in enumerate(level1_probs[0])
                }
            }
        }
        
        if level2_section:
            result['level2'] = {
                'subsection': level2_section,
                'confidence': level2_confidence,
                'full_path': f"{level1_section} > {level2_section}"
            }
        
        return result
    
    def analyze_document_structure(self, paragraphs: List[str]) -> Dict:
        """Analyzes the complete structure of a document"""
        document_structure = []
        section_transitions = []
        confidence_scores = []
        
        previous_section = None
        
        for i, paragraph in enumerate(paragraphs):
            prediction = self.predict_hierarchical(paragraph)
            current_section = prediction['level1']['section']
            
            document_structure.append({
                'paragraph_id': i,
                'text_preview': prediction['text_preview'],
                'section': current_section,
                'subsection': prediction.get('level2', {}).get('subsection'),
                'confidence': prediction['level1']['confidence']
            })
            
            confidence_scores.append(prediction['level1']['confidence'])
            
            # Detect transitions
            if previous_section and previous_section != current_section:
                section_transitions.append({
                    'position': i,
                    'from': previous_section,
                    'to': current_section
                })
            
            previous_section = current_section
        
        # Aggregated analysis
        section_counts = defaultdict(int)
        for item in document_structure:
            section_counts[item['section']] += 1
        
        return {
            'structure': document_structure,
            'transitions': section_transitions,
            'statistics': {
                'total_paragraphs': len(paragraphs),
                'unique_sections': len(set(item['section'] for item in document_structure)),
                'section_distribution': dict(section_counts),
                'avg_confidence': np.mean(confidence_scores),
                'min_confidence': min(confidence_scores),
                'coherence_score': self._calculate_coherence(section_transitions, len(paragraphs))
            }
        }
    
    def _calculate_coherence(self, transitions: List[Dict], total_paragraphs: int) -> float:
        """Calculates a coherence score based on transitions"""
        if total_paragraphs <= 1:
            return 1.0
        
        # Penalize excessive transitions
        expected_transitions = 8  # Typical number of main sections
        actual_transitions = len(transitions)
        
        coherence = max(0, 1 - abs(actual_transitions - expected_transitions) / total_paragraphs)
        return coherence
    
    def suggest_restructuring(self, analysis: Dict) -> List[str]:
        """Suggests improvements in document structure"""
        suggestions = []
        stats = analysis['statistics']
        structure = analysis['structure']
        
        # Get current sections in order
        current_sections = [item['section'] for item in structure]
        unique_sections = []
        for section in current_sections:
            if section not in unique_sections:
                unique_sections.append(section)
        
        # Suggestions based on analysis
        if stats['avg_confidence'] < 0.7:
            suggestions.append("⚠️ Low average classification confidence. "
                             "Consider reviewing section clarity and structure.")
        
        # Check for missing sections
        missing_critical = []
        for section in ['introduction', 'methods', 'results', 'conclusion']:
            if section not in unique_sections:
                missing_critical.append(section)
        
        if missing_critical:
            suggestions.append(f"📝 Missing critical sections: {', '.join(missing_critical)}")
        
        # Check logical order
        expected_order = ['abstract', 'introduction', 'related_work', 'methods', 
                         'results', 'discussion', 'conclusion', 'references']
        
        order_issues = []
        for i, expected in enumerate(expected_order):
            if expected in unique_sections:
                actual_pos = unique_sections.index(expected)
                if actual_pos != i and i < len(unique_sections):
                    # Check if significantly out of place
                    for j in range(i):
                        if expected_order[j] in unique_sections:
                            if unique_sections.index(expected_order[j]) > actual_pos:
                                order_issues.append(f"{expected} appears before {expected_order[j]}")
        
        if order_issues:
            suggestions.append(f"🔄 Ordering issues detected: {'; '.join(order_issues[:3])}")
        
        # Check coherence
        if stats['coherence_score'] < 0.6:
            suggestions.append("🔀 Too many section transitions. "
                             "Consider better grouping of related content.")
        
        # Check section balance
        section_dist = stats['section_distribution']
        total = sum(section_dist.values())
        for section, count in section_dist.items():
            proportion = count / total
            if section in ['methods', 'results'] and proportion < 0.15:
                suggestions.append(f"📊 The '{section}' section seems very brief ({proportion:.1%} of document). "
                                 "Consider expanding it.")
            elif section == 'introduction' and proportion > 0.3:
                suggestions.append(f"📏 The introduction is very extensive ({proportion:.1%} of document). "
                                 "Consider being more concise.")
        
        return suggestions if suggestions else ["✅ The document structure appears adequate."]


# ====================
# UTILITY FUNCTIONS
# ====================

def extract_features_for_classification(text: str, position_in_doc: float = 0.5) -> Dict:
    """
    Extracts additional features to improve classification.
    
    Args:
        text: Paragraph text
        position_in_doc: Relative position in document (0=start, 1=end)
    
    Returns:
        Dictionary with extracted features
    """
    features = {
        'length': len(text.split()),
        'position': position_in_doc,
        'has_citations': bool(re.search(r'\[\d+\]|\(\w+,?\s*\d{4}\)', text)),
        'has_numbers': bool(re.search(r'\d+\.?\d*', text)),
        'has_statistics': bool(re.search(r'p\s*[<>=]\s*0\.\d+|mean|median|SD|CI', text, re.IGNORECASE)),
        'has_figures': bool(re.search(r'Figure\s*\d+|Table\s*\d+|Fig\.\s*\d+', text, re.IGNORECASE)),
        'starts_with_number': bool(re.match(r'^\d+\.?\s+', text)),
        'has_keywords': {
            'introduction': any(word in text.lower() for word in 
                              ['introduction', 'background', 'motivation', 'overview']),
            'methods': any(word in text.lower() for word in 
                         ['method', 'procedure', 'algorithm', 'participants', 'data collection']),
            'results': any(word in text.lower() for word in 
                         ['results', 'findings', 'observed', 'showed', 'demonstrated']),
            'discussion': any(word in text.lower() for word in 
                           ['discussion', 'implications', 'limitations', 'interpretation']),
            'conclusion': any(word in text.lower() for word in 
                           ['conclusion', 'summary', 'future work', 'in conclusion'])
        }
    }
    
    return features


class EnhancedHierarchicalClassifier(HierarchicalSectionClassifier):
    """
    Enhanced version of the classifier with additional features
    and heuristic rules for greater accuracy.
    """
    
    def __init__(self, use_rules: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rules = use_rules
    
    def apply_heuristic_rules(self, text: str, ml_prediction: Dict, 
                             position: float = 0.5) -> Dict:
        """
        Applies heuristic rules to refine model predictions.
        """
        if not self.use_rules:
            return ml_prediction
        
        features = extract_features_for_classification(text, position)
        refined_prediction = ml_prediction.copy()
        
        # Rules for Abstract
        if position < 0.05 and features['length'] < 300:
            if 'objective' in text.lower() or 'aim' in text.lower():
                refined_prediction['level1']['section'] = 'abstract'
                refined_prediction['level1']['confidence'] = min(0.95, 
                    refined_prediction['level1']['confidence'] + 0.2)
        
        # Rules for References
        if features['has_citations'] and position > 0.9:
            citation_density = len(re.findall(r'\[\d+\]|\(\w+,?\s*\d{4}\)', text)) / features['length']
            if citation_density > 0.1:
                refined_prediction['level1']['section'] = 'references'
                refined_prediction['level1']['confidence'] = 0.99
        
        # Rules for Methods
        if features['has_keywords']['methods'] and 0.2 < position < 0.5:
            if refined_prediction['level1']['confidence'] < 0.7:
                refined_prediction['level1']['section'] = 'methods'
                refined_prediction['level1']['confidence'] = min(0.85,
                    refined_prediction['level1']['confidence'] + 0.15)
        
        # Rules for Results
        if features['has_statistics'] and features['has_figures']:
            if 0.4 < position < 0.7:
                refined_prediction['level1']['section'] = 'results'
                refined_prediction['level1']['confidence'] = min(0.90,
                    refined_prediction['level1']['confidence'] + 0.1)
        
        return refined_prediction
    
    def batch_classify(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Efficient batch classification for long documents.
        """
        results = []
        total_texts = len(texts)
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            batch_positions = [(i + j) / total_texts for j in range(len(batch))]
            
            for text, position in zip(batch, batch_positions):
                ml_prediction = self.predict_hierarchical(text)
                refined_prediction = self.apply_heuristic_rules(text, ml_prediction, position)
                results.append(refined_prediction)
        
        return results

# ============================================================
# ADDED: Hugging Face loader + train/eval (level-1 only)
# Expect dataset columns: 'text' and 'label' (required).
# Keeps all original classes/methods untouched.
# ============================================================

@dataclass
class HFArgs:  # ADDED
    dataset: str = 'json'  # 'json'/'csv'/'parquet' or HF hub name  # ADDED
    data_files: Optional[str] = None  # comma-separated file paths  # ADDED
    train_split: str = 'train'  # ADDED
    eval_split: str = 'validation'  # ADDED
    test_split: Optional[str] = 'test'  # ADDED
    model_name: str = 'allenai/scibert_scivocab_uncased'  # ADDED
    max_len: int = 256  # ADDED
    batch_size: int = 16  # ADDED
    epochs: int = 3  # ADDED
    lr: float = 2e-5  # ADDED
    weight_decay: float = 0.01  # ADDED
    warmup_ratio: float = 0.06  # ADDED
    gradient_accum: int = 1  # ADDED
    seed: int = 42  # ADDED
    output_dir: str = './outputs_hssc'  # ADDED
    amp: bool = True  # ADDED
    # --- NEW ---
    run_name: Optional[str] = None        # ADDED
    report_to: Optional[str] = None       # ADDED
    level2_col: Optional[str] = None
    l2_weight: float = 0.5

# ADDED (UPDATED): accept your .sh flags; auto-map aliases; no .sh changes needed
def _hf_parse_from_argv() -> Optional[HFArgs]:  # ADDED
    argv = sys.argv
    hf_mode = any(flag in argv for flag in ['--hf_train', '--do_train', '--dataset', '--dataset_name'])
    if not hf_mode:
        return None

    def _get(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    args = HFArgs()

    args.level2_col = _get('--level2_col', args.level2_col)
    args.l2_weight  = float(_get('--l2_weight', str(args.l2_weight)))

    # dataset / files
    args.dataset = (_get('--dataset', None) or _get('--dataset_name', None) or args.dataset)
    args.data_files = _get('--data_files', args.data_files)

    # splits
    args.train_split = _get('--train_split', args.train_split)
    args.eval_split  = _get('--eval_split', args.eval_split)
    args.test_split  = _get('--test_split', args.test_split)

    # model/tokenizer
    args.model_name = _get('--model_name', args.model_name)
    _ = _get('--tokenizer_name', None)  # accepted but unused

    # lengths / batch / epochs
    args.max_len    = int(_get('--max_len', _get('--max_length', str(args.max_len))))
    args.batch_size = int(_get('--batch_size', str(args.batch_size)))
    args.epochs     = int(_get('--epochs', _get('--num_train_epochs', str(args.epochs))))

    # optimizer
    args.lr            = float(_get('--lr', _get('--learning_rate', str(args.lr))))
    args.weight_decay  = float(_get('--weight_decay', str(args.weight_decay)))
    args.warmup_ratio  = float(_get('--warmup_ratio', str(args.warmup_ratio)))
    args.gradient_accum = int(_get('--gradient_accum', str(args.gradient_accum)))

    # misc
    args.seed       = int(_get('--seed', str(args.seed)))
    args.output_dir = _get('--output_dir', args.output_dir)

    # precision
    if '--no_amp' in argv:
        args.amp = False
    elif '--fp16' in argv:
        args.amp = True

    # --- NEW: W&B flags ---
    args.run_name  = _get('--run_name', args.run_name)       # ADDED
    args.report_to = _get('--report_to', args.report_to)     # ADDED

    return args

# ADDED (UPDATED): accept label_names + alias map, decode ints, alias to our keys
# ADDED (UPDATED): now optionally handles level2 as well
class HFLevel1Dataset(torch.utils.data.Dataset):  # ADDED
    """Maps dataset with columns 'text' (+ optional 'level2') onto indices."""
    def __init__(
        self,
        hf_split,
        tokenizer,
        level1_to_idx: Dict[str, int],
        max_len: int = 256,
        label_names: Optional[List[str]] = None,
        alias_map: Optional[Dict[str, str]] = None,
        level2_col: Optional[str] = None,                 # ADDED
        l2_to_idx_per_parent: Optional[Dict[str, Dict[str, int]]] = None  # ADDED
    ):
        self.data = hf_split
        self.tok = tokenizer
        self.l1_to_idx = level1_to_idx
        self.max_len = max_len
        self.label_names = label_names or []

        # canonical alias map (extend if needed)
        self.alias_map = {
            'intro': 'introduction',
            'introduction': 'introduction',
            'background': 'introduction',
            'related work': 'related_work', 'related_work': 'related_work',
            'materials_and_methods': 'methods', 'material_and_methods': 'methods',
            'methodology': 'methods', 'method': 'methods', 'methods': 'methods',
            'experiments_and_results': 'results', 'experiments': 'results', 'results': 'results',
            'results_and_discussion': 'discussion', 'discussion': 'discussion',
            'discussion_and_conclusion': 'conclusion', 'conclusions': 'conclusion', 'conclusion': 'conclusion',
            'acknowledgements': 'acknowledgments', 'acknowledgments': 'acknowledgments',
            'references': 'references', 'reference': 'references',
            'appendix': 'appendix', 'supplementary': 'appendix',
            'abstract': 'abstract', 'objective': 'abstract',
            'methods_summary': 'abstract', 'results_summary': 'abstract', 'conclusions_summary': 'abstract',
        }
        if alias_map:
            self.alias_map.update({k.lower(): v for k, v in alias_map.items()})

        # default fallback for known int labels (keeps working if ClassLabel.names missing)
        self.default_index_map = {0: 'introduction', 1: 'introduction', 2: 'methods', 3: 'results', 4: 'conclusion'}

        # ADDED: optional L2 wiring
        self.level2_col = level2_col
        self.l2_to_idx = l2_to_idx_per_parent or {}

        # basic schema check for L1 (we require text+label by your design)
        sample = hf_split[0]
        if 'text' not in sample or 'label' not in sample:
            raise ValueError("HuggingFace data must have 'text' and 'label' columns.")
        # L2 is optional; no error if missing

    def __len__(self): return self.data.num_rows

    def _decode_label(self, raw):
        # ints from ClassLabel -> name; if no names, use default_index_map
        if isinstance(raw, int):
            if 0 <= raw < len(self.label_names or []):
                name = self.label_names[raw]
            else:
                name = self.default_index_map.get(raw, str(raw))
        else:
            name = str(raw)
        key = name.strip().lower().replace(' ', '_').replace('-', '_')
        key = self.alias_map.get(key, key)
        return key

    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item['text'])
        # L1
        canon_l1 = self._decode_label(item['label'])
        l1 = self.l1_to_idx.get(canon_l1, -1)

        # tokenize
        enc = self.tok(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')

        out = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'level1': torch.tensor(l1, dtype=torch.long),
        }

        # ADDED: L2 (optional)
        if self.level2_col and self.level2_col in item:
            raw_l2 = item[self.level2_col]
            if raw_l2 is None:
                l2_idx = -1
            else:
                canon_l2 = str(raw_l2).strip().lower().replace(' ', '_').replace('-', '_')
                canon_l2 = self.alias_map.get(canon_l2, canon_l2)
                l2_idx = -1
                if canon_l1 in self.l2_to_idx:
                    l2_idx = self.l2_to_idx[canon_l1].get(canon_l2, -1)
            out['level2'] = torch.tensor(l2_idx, dtype=torch.long)

        return out

# ADDED: collate that tolerates optional 'level2'
def collate_fn(batch):
    import torch  # safe even if torch is already imported
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attn      = torch.stack([b['attention_mask'] for b in batch])
    l1        = torch.stack([b['level1'] for b in batch])
    out = {'input_ids': input_ids, 'attention_mask': attn, 'level1': l1}
    if 'level2' in batch[0]:
        l2 = torch.stack([b.get('level2', torch.tensor(-1, dtype=torch.long)) for b in batch])
        out['level2'] = l2
    return out

def _build_hf_dataloaders(args: HFArgs, clf: HierarchicalSectionClassifier):  # ADDED
    if load_dataset is None:
        raise RuntimeError("Please `pip install datasets` for HF loading.")
    # map files (unchanged) ...
    ds = load_dataset(args.dataset, data_files=data_files) if args.dataset in ('json','csv','parquet') else load_dataset(args.dataset)

    tr = ds[args.train_split]
    va = ds[args.eval_split] if args.eval_split in ds else None
    te = ds[args.test_split] if (args.test_split and args.test_split in ds) else None

    # ADDED: decode ClassLabel names if available
    def _label_names(split):
        try:
            feats = ds[split].features
            if 'label' in feats and hasattr(feats['label'], 'names') and feats['label'].names:
                return list(feats['label'].names)
        except Exception:
            pass
        return None

    ln_tr = _label_names(args.train_split)  # ADDED
    ln_va = _label_names(args.eval_split) if va is not None else None  # ADDED
    ln_te = _label_names(args.test_split) if te is not None else None  # ADDED

    tok = clf.tokenizer

    train_set = HFLevel1Dataset(tr, tok, clf.level1_to_idx, args.max_len, label_names=ln_tr, alias_map=None, level2_col=args.level2_col, l2_to_idx_per_parent=clf.level2_to_idx)
    val_set   = HFLevel1Dataset(va, tok, clf.level1_to_idx, args.max_len, label_names=ln_va, alias_map=None, level2_col=args.level2_col,l2_to_idx_per_parent=clf.level2_to_idx) if va is not None else None  # ADDED
    test_set  = HFLevel1Dataset(te, tok, clf.level1_to_idx, args.max_len, label_names=ln_te, alias_map=None, level2_col=args.level2_col,l2_to_idx_per_parent=clf.level2_to_idx) if te is not None else None  # ADDED

    # CHANGED: add collate_fn=collate_fn
    dl_tr = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  pin_memory=True,  collate_fn=collate_fn)   # CHANGED
    dl_va = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, pin_memory=True,  collate_fn=collate_fn) if val_set else None  # CHANGED
    dl_te = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, pin_memory=True,  collate_fn=collate_fn) if test_set else None  # CHANGED

    return dl_tr, dl_va, dl_te


def _set_seed(seed: int):  # ADDED
    import random  # ADDED
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)  # ADDED

# ADDED (UPDATED): return (Tensor loss, float acc)
def _compute_l1_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
    mask = labels >= 0
    if mask.sum() == 0:
        # scalar 0.0 tensor on same device for scaler/backward compatibility
        return logits.new_zeros((), requires_grad=True), 0.0
    loss_t = F.cross_entropy(logits[mask], labels[mask])
    preds = torch.argmax(logits, dim=1)
    acc = (preds[mask] == labels[mask]).float().mean().item()
    return loss_t, acc

# ADDED
def _compute_l2_loss(
    l2_logits_dict: Dict[str, torch.Tensor],
    l1_labels: torch.Tensor,
    l2_labels: torch.Tensor,
    idx_to_level1: Dict[int, str]
) -> Optional[torch.Tensor]:
    device = l1_labels.device
    N = l1_labels.size(0)
    losses = []
    for i in range(N):
        p = int(l1_labels[i])
        c = int(l2_labels[i])
        if p < 0 or c < 0: 
            continue
        parent = idx_to_level1[p]
        if parent not in l2_logits_dict:
            continue
        logits_i = l2_logits_dict[parent][i].unsqueeze(0)  # [1, C_parent]
        losses.append(F.cross_entropy(logits_i, torch.tensor([c], device=device)))
    if not losses:
        return None
    return torch.stack(losses).mean()

def hf_train_eval(args: HFArgs):  # ADDED
    os.makedirs(args.output_dir, exist_ok=True)
    _set_seed(args.seed)
    clf = HierarchicalSectionClassifier(model_name=args.model_name)
    model = clf.model
    device = clf.device
    dl_tr, dl_va, dl_te = _build_hf_dataloaders(args, clf)

    for pb in dl_tr:
        ys = pb['level1'].tolist()
        print(f"[Sanity] First batch mapped L1 indices (min={min(ys)}, max={max(ys)}):", ys[:16])
        break

    # ---------------- W&B init (respects your .sh env + flags) ----------------
    use_wandb = (args.report_to is not None) and ('wandb' in str(args.report_to).lower())  # ADDED
    if use_wandb and wandb is None:  # ADDED
        print("[WARN] report_to=wandb but wandb is not installed; continuing without logging.")  # ADDED
        use_wandb = False  # ADDED
    if use_wandb:  # ADDED
        wb_project = os.getenv('WANDB_PROJECT', 'hssc')  # ADDED
        wb_tags = [t for t in os.getenv('WANDB_TAGS', '').split(',') if t]  # ADDED
        wb_name = args.run_name or os.getenv('WANDB_RUN_NAME')  # ADDED
        wandb.init(project=wb_project, name=wb_name, tags=wb_tags, config=args.__dict__)  # ADDED
        try:  # ADDED
            wandb.watch(model, log='gradients', log_freq=100)  # ADDED
        except Exception:
            pass  # ADDED

    # ---------------- Optimizer & scheduler ----------------
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optim = AdamW(grouped, lr=args.lr)
    steps_per_epoch = max(1, len(dl_tr) // max(1, args.gradient_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())




    # ---------------- Training loop ----------------
    best_val = -1.0
    global_step = 0  # ADDED
    for ep in range(1, args.epochs+1):
        has_scaled = False  # ADDED: track if we called scaler.scale(...).backward() since the last step
        model.train()
        running_loss, running_acc, seen = 0.0, 0.0, 0
        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(dl_tr, 1):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attn = batch['attention_mask'].to(device, non_blocking=True)
            y1 = batch['level1'].to(device, non_blocking=True)


            with torch.amp.autocast('cuda', enabled=args.amp and torch.cuda.is_available()):
                # CHANGED (compute both; still works if no L2 available)
                l1_logits, l2_logits = model(input_ids, attn)
                # CHANGED: L1 loss (Tensor) + acc (float)
                l1_loss_t, l1_acc = _compute_l1_metrics(l1_logits, y1)

                l2_loss_t = None
                if 'level2' in batch and getattr(args, 'level2_col', None):
                    y2 = batch['level2'].to(device, non_blocking=True)
                    l2_loss_t = _compute_l2_loss(l2_logits, y1, y2, clf.idx_to_level1)  # returns Tensor or None

                # CHANGED: total loss for backprop
                total_loss = l1_loss_t if l2_loss_t is None else (l1_loss_t + args.l2_weight * l2_loss_t)



            # CHANGED: scale + backward with total_loss
            scaler.scale(total_loss / args.gradient_accum).backward()
            has_scaled = True  # keep

            # inside training loop:
            # ... after scaler.scale(loss_t / args.gradient_accum).backward()
            if step % args.gradient_accum == 0:
                # CHANGED: no unscale_(), guard scaler.step() with has_scaled and AMP enabled
                if scaler._enabled and has_scaled:
                    try:
                        scaler.step(optim)      # CHANGED
                        scaler.update()         # CHANGED
                    except AssertionError:
                        optim.step()            # CHANGED (rare fallback)
                else:
                    optim.step()                # CHANGED

                optim.zero_grad(set_to_none=True)
                sched.step()
                has_scaled = False              # ADDED: reset after stepping

            # CHANGED: running stats (floats for logs)
            running_loss += float(l1_loss_t.item())
            running_acc  += l1_acc
            seen += 1
            global_step += 1  # ADDED

            # ADDED: optional L2 running stats for logging (don’t affect training if None)
            if 'l2_running' not in locals():  # init once
                l2_running, l2_seen = 0.0, 0
            if l2_loss_t is not None:
                l2_running += float(l2_loss_t.item())
                l2_seen += 1
                
            # CHANGED: periodic console + W&B logs
            if step % 20 == 0 or step == len(dl_tr):
                curr_lr = optim.param_groups[0]['lr']
                msg = f"[Epoch {ep} | {step}/{len(dl_tr)}] L1_loss={running_loss/seen:.4f} L1_acc={running_acc/seen:.4f}"
                if l2_seen > 0:  # ADDED
                    msg += f" L2_loss={l2_running/l2_seen:.4f}"
                print(msg)

                if use_wandb:  # ADDED
                    log_dict = {
                        "train/l1_loss": running_loss/seen,
                        "train/l1_acc":  running_acc/seen,
                        "train/lr":      curr_lr,
                        "epoch":         ep,
                        "step":          global_step
                    }
                    if l2_seen > 0:  # ADDED
                        log_dict["train/l2_loss"] = l2_running/l2_seen
                    wandb.log(log_dict)

        # ---------------- Validation ----------------
        val_score = -1.0
        if dl_va is not None:
            model.eval()
            v_l, v_a, v_seen = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in dl_va:
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attn = batch['attention_mask'].to(device, non_blocking=True)
                    y = batch['level1'].to(device, non_blocking=True)
                    # CHANGED: get both levels
                    l1_logits, l2_logits = model(input_ids, attn)
                    l1_loss_t, l1_acc = _compute_l1_metrics(l1_logits, y)

                    # ADDED: optional L2 validation loss
                    l2_loss_t = None
                    if 'level2' in batch and getattr(args, 'level2_col', None):
                        # here y is your level1; fetch level2 from batch if builder provided it
                        y2 = batch['level2'].to(device, non_blocking=True)
                        l2_loss_t = _compute_l2_loss(l2_logits, y, y2, clf.idx_to_level1)

                    v_l += float(l1_loss_t.item())
                    v_a += l1_acc
                    v_seen += 1
                    if 'v_l2' not in locals():  # ADDED init once per epoch
                        v_l2, v_l2_seen = 0.0, 0
                    if l2_loss_t is not None:   # ADDED
                        v_l2 += float(l2_loss_t.item())
                        v_l2_seen += 1

            v_l /= max(1, v_seen); v_a /= max(1, v_seen)
            val_score = v_a

            # CHANGED: after averaging v_l, v_a
            val_msg = f"[Validation] L1_loss={v_l:.4f} L1_acc={v_a:.4f}"
            if 'v_l2_seen' in locals() and v_l2_seen > 0:  # ADDED
                val_msg += f" L2_loss={v_l2/v_l2_seen:.4f}"
            print(val_msg)

            if use_wandb:  # ADDED
                log_dict = {"val/l1_loss": v_l, "val/l1_acc": v_a, "epoch": ep, "step": global_step}
                if 'v_l2_seen' in locals() and v_l2_seen > 0:
                    log_dict["val/l2_loss"] = v_l2 / v_l2_seen
                wandb.log(log_dict)

        # ---------------- Checkpoints ----------------
        import os as _os
        ckpt = _os.path.join(args.output_dir, f'ep{ep}.pt')
        torch.save({'model': model.state_dict(), 'hf_args': args.__dict__}, ckpt)
        print(f"✓ Saved checkpoint: {ckpt}")
        if use_wandb:  # ADDED
            try:
                wandb.log({"checkpoint_path": ckpt, "epoch": ep, "step": global_step})
            except Exception:
                pass

        if val_score >= best_val:
            best_val = val_score
            best = _os.path.join(args.output_dir, 'best.pt')
            torch.save({'model': model.state_dict(), 'hf_args': args.__dict__}, best)
            print(f"✓ Updated best: {best}")
            if use_wandb:  # ADDED
                wandb.summary["best/val_acc"] = best_val

    # ---------------- Test (optional) ----------------
    if dl_te is not None:
        model.eval()
        t_l, t_a, t_seen = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in dl_te:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attn = batch['attention_mask'].to(device, non_blocking=True)
                y = batch['level1'].to(device, non_blocking=True)
                l1_logits, _ = model(input_ids, attn)
                loss_t, acc = _compute_l1_metrics(l1_logits, y)
                t_l += float(loss_t.item()); t_a += acc; t_seen += 1
        t_l /= max(1, t_seen); t_a /= max(1, t_seen)
        print(f"[Test] L1_loss={t_l:.4f} L1_acc={t_a:.4f}")
        if use_wandb:  # ADDED
            wandb.log({"test/loss": t_l, "test/acc": t_a})
            wandb.summary["test/acc"] = t_a
            try:
                wandb.finish()
            except Exception:
                pass
    else:
        if use_wandb:  # ADDED
            try:
                wandb.finish()
            except Exception:
                pass


def run_hf_from_cli():  # ADDED
    args = _hf_parse_from_argv()  # ADDED
    if args is None:  # ADDED
        return False  # ADDED
    print("[HSSC:HF] Parsed args:", {k: v for k, v in args.__dict__.items()})  # ADDED
    hf_train_eval(args)  # ADDED
    return True  # ADDED


# ====================
# USAGE EXAMPLE
# ====================

def demo_hierarchical_classification():
    """Demonstration of the hierarchical classification system"""
    
    print("=" * 70)
    print("HIERARCHICAL SCIENTIFIC SECTION CLASSIFICATION SYSTEM")
    print("=" * 70)
    
    # Initialize classifier
    print("\n→ Initializing hierarchical classifier...")
    classifier = EnhancedHierarchicalClassifier(use_rules=True)
    
    # Sample document
    sample_document = [
        "This study examines the impact of artificial intelligence on healthcare outcomes. "
        "We analyzed data from 500 hospitals over a 5-year period to assess the effectiveness "
        "of AI-assisted diagnosis systems.",
        
        "Artificial intelligence has revolutionized many aspects of modern medicine. "
        "Previous studies have shown promising results in radiology and pathology applications. "
        "However, comprehensive analyses of system-wide impacts remain limited.",
        
        "We conducted a retrospective cohort study using electronic health records. "
        "Hospitals were categorized based on their level of AI implementation. "
        "Primary outcomes included diagnostic accuracy, time to diagnosis, and patient satisfaction.",
        
        "Statistical analysis was performed using Python 3.8 with scikit-learn. "
        "We employed propensity score matching to control for confounding variables. "
        "Significance was set at p < 0.05 for all tests.",
        
        "AI-assisted hospitals showed a 23% improvement in diagnostic accuracy (p < 0.001). "
        "The mean time to diagnosis decreased from 4.2 to 2.8 days. "
        "Patient satisfaction scores increased by 18 points on a 100-point scale.",
        
        "These findings suggest that AI integration can substantially improve healthcare delivery. "
        "The observed improvements were most pronounced in complex cases requiring multi-specialty consultation. "
        "However, implementation costs and training requirements remain significant barriers.",
        
        "Several limitations should be considered when interpreting these results. "
        "First, the study was retrospective and subject to selection bias. "
        "Second, AI systems varied considerably across institutions.",
        
        "In conclusion, this large-scale analysis demonstrates the potential of AI in healthcare. "
        "Future research should focus on standardization and cost-effectiveness. "
        "Prospective trials are needed to confirm these findings."
    ]
    
    print(f"✓ Document loaded with {len(sample_document)} paragraphs\n")
    
    # Document structure analysis
    print("→ Analyzing document structure...")
    analysis = classifier.analyze_document_structure(sample_document)
    
    print("\n" + "=" * 70)
    print("DETECTED STRUCTURE")
    print("=" * 70)
    
    for item in analysis['structure']:
        section = item['section'].upper()
        subsection = f" > {item['subsection']}" if item['subsection'] else ""
        confidence = item['confidence']
        
        print(f"\nParagraph {item['paragraph_id'] + 1}:")
        print(f"  Section: {section}{subsection}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Text: \"{item['text_preview'][:80]}...\"")
    
    # Statistics
    print("\n" + "=" * 70)
    print("DOCUMENT STATISTICS")
    print("=" * 70)
    
    stats = analysis['statistics']
    print(f"Summary:")
    print(f"  • Total paragraphs: {stats['total_paragraphs']}")
    print(f"  • Unique sections: {stats['unique_sections']}")
    print(f"  • Average confidence: {stats['avg_confidence']:.2%}")
    print(f"  • Coherence score: {stats['coherence_score']:.2f}/1.00")
    
    print(f"Section distribution:")
    for section, count in sorted(stats['section_distribution'].items(), 
                                 key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_paragraphs']) * 100
        bar = '█' * int(percentage / 5)
        print(f"  • {section:15s}: {count:2d} paragraphs ({percentage:5.1f}%) {bar}")
    
    # Transitions
    if analysis['transitions']:
        print(f"Detected transitions:")
        for trans in analysis['transitions']:
            print(f"  • Paragraph {trans['position']}: {trans['from']} → {trans['to']}")
    
    # Improvement suggestions
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 70)
    
    suggestions = classifier.suggest_restructuring(analysis)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion}")
    
    # Example of classification with heuristic rules
    print("\n" + "=" * 70)
    print("COMPARISON: ML vs ML+RULES")
    print("=" * 70)
    
    test_text = ("The results demonstrate a significant improvement in performance metrics. "
                 "Table 2 shows the comparison between baseline and our proposed method. "
                 "Statistical significance was assessed using paired t-tests (p < 0.01).")
    
    # Without rules
    classifier_no_rules = HierarchicalSectionClassifier()
    pred_no_rules = classifier_no_rules.predict_hierarchical(test_text)
    
    # With rules
    pred_with_rules = classifier.apply_heuristic_rules(
        test_text, 
        pred_no_rules, 
        position=0.5
    )
    
    print(f"\nText: \"{test_text[:100]}...\"")
    print(f"\n  Pure ML prediction:")
    print(f"    Section: {pred_no_rules['level1']['section']}")
    print(f"    Confidence: {pred_no_rules['level1']['confidence']:.2%}")
    
    print(f"\n  ML + Rules prediction:")
    print(f"    Section: {pred_with_rules['level1']['section']}")
    print(f"    Confidence: {pred_with_rules['level1']['confidence']:.2%}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETED")
    print("=" * 70)


# CHANGED: main entry — keeps original demo as default, but if --hf_train/--hf_eval
# are provided, runs HuggingFace training/eval on 'text' + 'label' data.
if __name__ == "__main__":
    if not run_hf_from_cli():  # ADDED
        demo_hierarchical_classification()  # original default
