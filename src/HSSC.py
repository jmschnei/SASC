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
        self.bert = AutoModel.from_pretrained(model_name)
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


if __name__ == "__main__":
    demo_hierarchical_classification()