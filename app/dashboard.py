import matplotlib.pyplot as plt
import io
import base64
from src.utils.logger import app_logger

def generate_analytics():
    """Generate analytics data for dashboard"""
    logger = app_logger
    logger.info("Generating analytics data")
    
    # Sample data - in a real application, this would come from a database
    intent_distribution = {
        'refund': 40,
        'product_info': 30,
        'technical_support': 20,
        'billing': 10
    }
    
    # Create intent distribution chart
    plt.figure(figsize=(8, 6))
    plt.pie(intent_distribution.values(), labels=intent_distribution.keys(), autopct='%1.1f%%')
    plt.title('Intent Distribution')
    intent_chart = plot_to_base64()
    
    # Create query trends chart
    plt.figure(figsize=(10, 6))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    queries = [120, 150, 130, 140, 160, 100, 80]
    plt.plot(days, queries, marker='o')
    plt.title('Daily Query Volume')
    plt.xlabel('Day')
    plt.ylabel('Number of Queries')
    plt.grid(True)
    trend_chart = plot_to_base64()
    
    # Prepare analytics data
    analytics_data = {
        'total_queries': 880,
        'accuracy': 87.5,
        'avg_response_time': 1.2,
        'fallback_rate': 12.5,
        'intent_chart': intent_chart,
        'trend_chart': trend_chart
    }
    
    return analytics_data

def plot_to_base64():
    """Convert matplotlib plot to base64 encoded image"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return image_base64