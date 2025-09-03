import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report
from src.chatbot.chatbot_core import CustomerSupportChatbot
from src.utils.logger import app_logger

def main():
    logger = app_logger
    logger.info("Starting model evaluation")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize chatbot
    chatbot = CustomerSupportChatbot(config)
    
    # Test data (in a real application, this would be your test dataset)
    test_data = [
        ("How do I get a refund?", "refund"),
        ("What's your return policy?", "refund"),
        ("Tell me about your products", "product_info"),
        ("When are you open?", "hours"),
        ("My device isn't working", "technical"),
        ("Need help with setup", "technical"),
        ("Where's my order?", "tracking"),
        ("When will it arrive?", "delivery"),
        ("Can I change delivery address?", "shipping"),
        ("Do you offer warranty?", "warranty"),
        ("What payment methods?", "payment")
    ]
    
    # Evaluate intent classification
    true_labels = []
    predicted_labels = []
    
    for query, true_intent in test_data:
        predicted_intent, confidence = chatbot.intent_classifier.predict(query)
        true_labels.append(true_intent)
        predicted_labels.append(predicted_intent)
        logger.info(f"Query: {query} | True: {true_intent} | Predicted: {predicted_intent} | Confidence: {confidence:.4f}")
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    logger.info(f"Intent Classification Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(true_labels, predicted_labels)
    logger.info("Classification Report:\n" + report)
    
    # Test response generation
    logger.info("Testing response generation:")
    for query, _ in test_data[:3]:  # Test first 3 queries
        response = chatbot.process_message(query)
        logger.info(f"Query: {query} | Response: {response}")
    
    logger.info("Model evaluation completed")

if __name__ == "__main__":
    main()