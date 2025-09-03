import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import app_logger

class IntentLabeler:
    def __init__(self):
        self.logger = app_logger
        self.label_encoder = LabelEncoder()
        self.intent_mapping = {}
    
    def fit(self, intents: list):
        """Fit label encoder to intent list"""
        self.label_encoder.fit(intents)
        self.intent_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        self.logger.info(f"Fitted label encoder with {len(self.intent_mapping)} intents")
    
    def transform(self, intents: list) -> list:
        """Transform intent labels to encoded values"""
        return self.label_encoder.transform(intents)
    
    def inverse_transform(self, encoded_labels: list) -> list:
        """Transform encoded labels back to original intent names"""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_intent_mapping(self) -> dict:
        """Get intent mapping dictionary"""
        return self.intent_mapping
    
    def save_mapping(self, file_path: str):
        """Save intent mapping to file"""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.intent_mapping, f, indent=2)
        self.logger.info(f"Intent mapping saved to {file_path}")
    
    def load_mapping(self, file_path: str):
        """Load intent mapping from file"""
        import json
        with open(file_path, 'r') as f:
            self.intent_mapping = json.load(f)
        self.logger.info(f"Intent mapping loaded from {file_path}")