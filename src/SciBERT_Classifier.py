"""
SASC using SciBERT
==================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


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


class SciBERTSectionClassifier(nn.Module):
    """Clasifier based on SciBERT for scientific section classification"""
    
    def __init__(self, n_classes: int, model_name: str = 'allenai/scibert_scivocab_uncased'):
        super(SciBERTSectionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
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
        'references': 7,
        'acknowledgments': 8,
        'supplementary': 9
    }
    
    def __init__(self, model_name: str = 'allenai/scibert_scivocab_uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_to_section = {v: k for k, v in self.SECTION_LABELS.items()}
        
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
              epochs: int = 5, learning_rate: float = 2e-5) -> Dict:
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
            train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Store metrics
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print("-" * 50)
        
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
    
    print("=" * 60)
    print("SISTEMA DE CLASIFICACIÓN DE SECCIONES CIENTÍFICAS")
    print("=" * 60)
    
    # Generar datos de ejemplo
    texts, labels = generate_sample_data()
    print(f"\n✓ Datos cargados: {len(texts)} ejemplos")
    
    # Inicializar sistema
    print("\n→ Inicializando modelo SciBERT...")
    classifier = SectionClassificationSystem()
    
    # Preparar datos
    texts_processed, label_ids = classifier.prepare_data(texts, labels)
    print(f"✓ Datos procesados: {len(texts_processed)} ejemplos válidos")
    
    # Crear DataLoaders
    train_loader, val_loader = classifier.create_data_loaders(
        texts_processed, label_ids, test_size=0.3, batch_size=4
    )
    print(f"✓ DataLoaders creados")
    
    # Entrenar modelo (reducido para demostración)
    print("\n→ Entrenando modelo (modo demo - 2 epochs)...")
    print("-" * 50)
    history = classifier.train(train_loader, val_loader, epochs=2)
    
    # Realizar predicciones de ejemplo
    print("\n" + "=" * 60)
    print("PREDICCIONS OF EXAMPLE")
    print("=" * 60)
    
    test_texts = [
        "The objective of this research is to explore new methods in artificial intelligence.",
        "Data was collected from 100 participants using online surveys.",
        "The results indicate a strong correlation between the variables studied."
    ]
    
    predictions = classifier.predict(test_texts)
    
    for pred in predictions:
        print(f"\nTexto: '{pred['text']}'")
        print(f"Sección predicha: {pred['predicted_section'].upper()}")
        print(f"Confianza: {pred['confidence']:.2%}")
        print("Probabilidades por sección:")
        for section, prob in sorted(pred['all_probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]:
            print(f"  - {section}: {prob:.2%}")
    
    print("\n" + "=" * 60)
    print("Ready to use")
    print("=" * 60)


# Execute main script
if __name__ == "__main__":
    main()