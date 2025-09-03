# AI-Powered Customer Support Chatbot

An intelligent chatbot for customer support using NLP and deep learning.

## Features
- Intent classification with BERT
- Response generation with seq2seq model
- Fallback mechanism for unknown queries
- Web interface for interaction
- Analytics dashboard

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download spaCy model: `python -m spacy download en_core_web_sm`
4. Train models: `python scripts/train_model.py`
5. Run the app: `python app.py`

## Usage
Access the web interface at http://localhost:5000
Use the API endpoint `/chat` for programmatic access

## Project Structure
- `src/`: Core chatbot functionality
- `app/`: Web application components
- `data/`: Raw and processed data
- `models/`: Trained model files
- `notebooks/`: Jupyter notebooks for exploration
- `tests/`: Unit tests
- `config/`: Configuration files
- `scripts/`: Utility scripts

## API Endpoints
- `POST /chat`: Send a message to the chatbot
- `GET /dashboard`: View analytics dashboard

## Deployment
The application can be deployed using Docker:
```bash
docker build -t customer-support-chatbot .
docker run -p 5000:5000 customer-support-chatbot