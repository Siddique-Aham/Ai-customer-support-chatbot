from .intent_classifier import IntentClassifier
from .response_generator import ResponseGenerator
from .fallback_handler import FallbackHandler
from src.utils.config import get_model_paths
from src.utils.logger import app_logger



class CustomerSupportChatbot:
    def __init__(self, config):
        self.logger = app_logger
        self.config = config
        
        # Get model paths
        model_paths = get_model_paths(config)
        
        # Initialize components
        self.intent_classifier = IntentClassifier(
            model_paths['intent_classifier'],
            config
        )
        
        self.response_generator = ResponseGenerator(
            model_paths['response_generator'],
            config
        )
        
        self.fallback_handler = FallbackHandler(
            model_paths['fallback'],
            config
        )
        
        self.logger.info("Customer Support Chatbot initialized")
    
    def process_message(self, message):
        """Process user message and generate response"""
        try:
            # Classify intent
            intent, confidence = self.intent_classifier.predict(
                message,
                self.config['chatbot']['confidence_threshold']
            )
            
            self.logger.info(f"Intent: {intent}, Confidence: {confidence:.4f}")
            
            # Generate response based on intent
            if intent != "unknown":
                response = self.response_generator.generate_response(
                    message,
                    intent,
                    self.config['chatbot']['max_response_length']
                )
            else:
                # Use fallback for unknown intents
                response = self.fallback_handler.handle_unknown_query(
                    message,
                    self.config['chatbot']['fallback_threshold']
                )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."