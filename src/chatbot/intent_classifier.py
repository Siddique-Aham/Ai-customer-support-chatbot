import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from src.utils.logger import app_logger

class IntentClassifier:
    def __init__(self, model_path=None, config=None):
        self.logger = app_logger
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path:
            self.load_model(model_path)
        else:
            self.model = None
            self.tokenizer = None
    
    def load_model(self, model_path):
        """Load pre-trained intent classification model"""
        try:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Intent classification model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, text, threshold=0.7):
        """Predict intent for given text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        confidence = confidence.item()
        predicted_label = predicted.item()
        
        # Apply confidence threshold
        if confidence < threshold:
            return "unknown", confidence
        
        return predicted_label, confidence