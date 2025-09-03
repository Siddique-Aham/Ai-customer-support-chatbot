import nltk
import spacy
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from src.utils.logger import app_logger

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.logger = app_logger
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            self.logger.warning("spaCy model not found. Using NLTK for preprocessing.")
            self.nlp = None
    
    def tokenize(self, text: str) -> list:
        """Tokenize text using spaCy or NLTK"""
        if self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc]
        else:
            return word_tokenize(text)
    
    def remove_stopwords(self, tokens: list) -> list:
        """Remove stopwords from tokens"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: list) -> list:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str) -> str:
        """Full text preprocessing pipeline"""
        # Tokenize
        tokens = self.tokenize(text)
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        # Lemmatize
        tokens = self.lemmatize(tokens)
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Preprocess entire dataset"""
        self.logger.info("Preprocessing dataset")
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Preprocess text column
        processed_df[text_column] = processed_df[text_column].apply(self.preprocess_text)
        
        self.logger.info("Dataset preprocessing completed")
        return processed_df