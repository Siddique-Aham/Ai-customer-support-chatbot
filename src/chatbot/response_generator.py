from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import tensorflow as tf
from src.utils.logger import app_logger

class ResponseGenerator:
    def __init__(self, model_path=None, config=None):
        self.logger = app_logger
        self.config = config or {}
        
        if model_path:
            self.load_model(model_path)
        else:
            self.model = None
            self.tokenizer = None
    
    def load_model(self, model_path):
        """Load pre-trained response generation model"""
        try:
            self.model = TFT5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.logger.info(f"Response generation model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(self, query, intent=None, max_length=50):
        """Generate response for given query and intent"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Prepare input text
        if intent:
            input_text = f"intent: {intent} query: {query}"
        else:
            input_text = query
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors='tf',
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # Generate response
        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response