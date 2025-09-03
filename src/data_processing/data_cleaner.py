import pandas as pd
import re
from src.utils.logger import app_logger

class DataCleaner:
    def __init__(self):
        self.logger = app_logger
    
    def clean_text(self, text: str) -> str:
        """Clean text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def clean_dataset(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Clean entire dataset"""
        self.logger.info("Cleaning dataset")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Clean text column
        cleaned_df[text_column] = cleaned_df[text_column].apply(self.clean_text)
        
        # Remove empty rows
        cleaned_df = cleaned_df[cleaned_df[text_column].str.len() > 0]
        
        self.logger.info(f"Dataset cleaned. Original: {len(df)} rows, Cleaned: {len(cleaned_df)} rows")
        return cleaned_df