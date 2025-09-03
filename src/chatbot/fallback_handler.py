from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import os
import random
from src.utils.logger import app_logger

class FallbackHandler:
    def __init__(self, model_path=None, config=None):
        self.logger = app_logger
        self.config = config or {}
        self.vectorizer = None
        self.known_queries = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load pre-trained fallback model"""
        try:
            # Load vectorizer
            with open(os.path.join(model_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load known queries
            self.known_queries = pd.read_csv(os.path.join(model_path, 'known_queries.csv'))
            
            self.logger.info(f"Fallback model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading fallback model: {e}")
            raise
    
    def handle_unknown_query(self, query, threshold=0.4):
        """Handle unknown queries with more confident responses"""
        try:
            if self.vectorizer is None or self.known_queries is None:
                return "I'd be happy to help with that! Please contact our support team for assistance."
            
            # Transform query to TF-IDF vector
            query_vec = self.vectorizer.transform([query])
            
            # Transform known queries to TF-IDF vectors
            known_vecs = self.vectorizer.transform(self.known_queries['query'])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vec, known_vecs).flatten()
            
            # Find most similar query
            max_similarity = similarities.max()
            max_index = similarities.argmax()
            
            if max_similarity >= threshold:
                response = self.known_queries.iloc[max_index]['response']
                return f"I understand you're asking about: {response}"
            else:
                # More confident generic responses
                responses = [
                    "I'd be happy to help with that!",
                    "I can assist you with that!",
                    "Let me help you with that.",
                    "I understand you're asking about this."
                ]
                response = random.choice(responses)
                response += " You can check our help center or contact support for more specific assistance."
                return response
                
        except Exception as e:
            self.logger.error(f"Error in fallback handler: {e}")
            return "I'll help you with that. Please visit our help center or contact support for more details."