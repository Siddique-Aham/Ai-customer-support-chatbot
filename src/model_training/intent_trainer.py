import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os  # ADD THIS IMPORT
from src.utils.logger import app_logger

class IntentTrainer:
    def __init__(self, config):
        self.logger = app_logger
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
    
    def prepare_data(self, texts, labels):
        """Prepare data for BERT training"""
        self.logger.info("Preparing data for intent classification")
        
        # DEBUG: Check labels type
        self.logger.info(f"Labels received: {labels[:5]}, Type: {type(labels[0])}")
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Convert labels to tensor (ensure they are integers)
        # CRITICAL FIX: Convert to numpy array first, then to tensor
        labels_array = np.array(labels, dtype=np.int64)
        labels_tensor = torch.tensor(labels_array).long()
        
        self.logger.info(f"Labels converted to tensor: {labels_tensor[:5]}")
        
        return TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels_tensor)
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Train intent classification model"""
        self.logger.info("Training intent classification model")
        
        # SAFE CONFIG ACCESS
        batch_size = self.config.get('training', {}).get('batch_size', 16)
        epochs = self.config.get('training', {}).get('epochs', 10)
        learning_rate = float(self.config.get('training', {}).get('learning_rate', 2e-5))  # FIXED LINE
        
        # DEBUG: Check unique labels
        unique_labels = list(set(train_labels))
        self.logger.info(f"Unique labels: {unique_labels}, Count: {len(unique_labels)}")
        
        # Prepare training data
        train_dataset = self.prepare_data(train_texts, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data if provided
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = self.prepare_data(val_texts, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(unique_labels)
        )
        self.model.to(self.device)
        
        # Set up optimizer - FIXED
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                # Log progress
                if batch_idx % 2 == 0:
                    self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        self.logger.info("Intent classification model training completed")
        return self.model
    
    def save_model(self, model_path):
        """Save trained model"""
        if self.model:
            # CRITICAL FIX: Ensure directory exists
            os.makedirs(model_path, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            
            # Also save config for loading
            model_config = {
                'num_labels': self.model.config.num_labels,
                'id2label': {i: f"intent_{i}" for i in range(self.model.config.num_labels)},
                'label2id': {f"intent_{i}": i for i in range(self.model.config.num_labels)}
            }
            
            import json
            with open(os.path.join(model_path, 'config.json'), 'w') as f:
                json.dump(model_config, f, indent=2)
            
            self.logger.info(f"Model saved to {model_path}")
            self.logger.info(f"Files created: {os.listdir(model_path)}")