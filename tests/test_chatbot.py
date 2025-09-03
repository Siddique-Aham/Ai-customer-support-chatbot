import unittest
from src.chatbot.chatbot_core import CustomerSupportChatbot
import yaml

class TestChatbot(unittest.TestCase):
    def setUp(self):
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        self.chatbot = CustomerSupportChatbot(self.config)
    
    def test_process_message(self):
        response = self.chatbot.process_message("How do I get a refund?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_unknown_intent(self):
        response = self.chatbot.process_message("Random gibberish text")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

if __name__ == '__main__':
    unittest.main()