import os
import subprocess
import yaml
from src.utils.logger import app_logger

def main():
    logger = app_logger
    logger.info("Starting deployment process")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Build Docker image
    try:
        logger.info("Building Docker image...")
        subprocess.run([
            'docker', 'build', 
            '-t', 'customer-support-chatbot',
            '.'
        ], check=True)
        logger.info("Docker image built successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building Docker image: {e}")
        return
    
    # Run Docker container
    try:
        logger.info("Starting Docker container...")
        subprocess.run([
            'docker', 'run', 
            '-d',
            '-p', f"{config['app']['port']}:5000",
            '--name', 'chatbot-container',
            'customer-support-chatbot'
        ], check=True)
        logger.info(f"Chatbot deployed and running on port {config['app']['port']}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Docker container: {e}")
        return
    
    logger.info("Deployment completed successfully")

if __name__ == "__main__":
    main()