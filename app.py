import os
from flask import Flask, render_template, request, jsonify
from pathlib import Path

# ✅ Correct import from config.py
from src.utils.config import load_config

# Load configuration
config = load_config()

# Initialize Flask app
app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(__file__), 'app', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'app', 'static')
)

# Import and initialize chatbot
from src.chatbot.chatbot_core import CustomerSupportChatbot
chatbot = CustomerSupportChatbot(config)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    data = {
        'total_queries': 150,
        'accuracy': 85.7,
        'avg_response_time': 1.2,
        'fallback_rate': 12.3,
        'intent_chart': '',
        'trend_chart': ''
    }
    return render_template('dashboard.html', data=data)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        response = chatbot.process_message(user_message)  # ✅ process_message use karo
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)