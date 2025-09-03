import unittest
import pandas as pd
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.preprocessor import TextPreprocessor

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.cleaner = DataCleaner()
        self.preprocessor = TextPreprocessor()
        self.sample_text = "Hello! Check out our website: https://example.com @username #hashtag"
    
    def test_clean_text(self):
        cleaned = self.cleaner.clean_text(self.sample_text)
        self.assertNotIn("https://example.com", cleaned)
        self.assertNotIn("@username", cleaned)
        self.assertNotIn("#hashtag", cleaned)
    
    def test_preprocess_text(self):
        processed = self.preprocessor.preprocess_text("I am running quickly")
        self.assertIn("run", processed)  # Should be lemmatized
    
    def test_dataset_cleaning(self):
        df = pd.DataFrame({'text': [self.sample_text, "Normal text"]})
        cleaned_df = self.cleaner.clean_dataset(df, 'text')
        self.assertEqual(len(cleaned_df), 2)
    
    def test_dataset_preprocessing(self):
        df = pd.DataFrame({'text': ["I am running", "You are walking"]})
        processed_df = self.preprocessor.preprocess_dataset(df, 'text')
        self.assertEqual(len(processed_df), 2)

if __name__ == '__main__':
    unittest.main()