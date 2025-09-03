import yaml
import os

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model_paths(config):
    """Get paths for all model directories"""
    models_config = config['models']
    return {
        'intent_classifier': models_config['intent_classifier_path'],
        'response_generator': models_config['response_generator_path'],
        'fallback': models_config['fallback_path']
    }