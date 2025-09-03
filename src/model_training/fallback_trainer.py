from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import os
from src.utils.logger import app_logger

class FallbackTrainer:
    def __init__(self, config):
        self.logger = app_logger
        self.config = config
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.known_queries = None
    
    def train(self, queries, responses):
        """Train fallback mechanism"""
        self.logger.info("Training fallback mechanism")
        
        # Store known queries and responses
        self.known_queries = pd.DataFrame({
            'query': queries,
            'response': responses
        })
        
        # Fit TF-IDF vectorizer
        self.vectorizer.fit(queries)
        
        self.logger.info("Fallback mechanism training completed")
    
    def find_similar_query(self, query, threshold=0.4):
        """Find similar query using cosine similarity"""
        if self.known_queries is None:
            return None, 0
        
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
            return self.known_queries.iloc[max_index]['response'], max_similarity
        else:
            return None, max_similarity
    
    def save_model(self, model_path):
        """Save trained fallback model"""
        os.makedirs(model_path, exist_ok=True)
        
        # Save vectorizer
        with open(os.path.join(model_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save known queries
        self.known_queries.to_csv(os.path.join(model_path, 'known_queries.csv'), index=False)
        
        self.logger.info(f"Fallback model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load trained fallback model"""
        # Load vectorizer
        with open(os.path.join(model_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load known queries
        self.known_queries = pd.read_csv(os.path.join(model_path, 'known_queries.csv'))
        
        self.logger.info(f"Fallback model loaded from {model_path}")