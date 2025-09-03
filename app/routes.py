from flask import render_template, request, jsonify
from src.chatbot.chatbot_core import CustomerSupportChatbot
from app.dashboard import generate_analytics
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize chatbot
chatbot = CustomerSupportChatbot(config)

def init_routes(app):
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/chat', methods=['POST'])
    def chat():
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        response = chatbot.process_message(user_message)
        return jsonify({'response': response})
    
    @app.route('/dashboard')
    def dashboard():
        analytics_data = generate_analytics()
        return render_template('dashboard.html', data=analytics_data)