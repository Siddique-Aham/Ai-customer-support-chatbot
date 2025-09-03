import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from sklearn.model_selection import train_test_split
import numpy as np
from src.utils.logger import app_logger

class ResponseTrainer:
    def __init__(self, config):
        self.logger = app_logger
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = None
    
    def prepare_data(self, queries, responses):
        """Prepare data for response generation training"""
        self.logger.info("Preparing data for response generation")
        
        # Tokenize inputs
        input_encodings = self.tokenizer(
            queries.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='tf'
        )
        
        # Tokenize targets
        target_encodings = self.tokenizer(
            text_target=responses.tolist(),
            truncation=True,
            padding=True,
            max_length=self.config['chatbot']['max_response_length'],
            return_tensors='tf'
        )
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'decoder_input_ids': target_encodings['input_ids'],
            'labels': target_encodings['input_ids']  # T5 expects labels for training
        }
    
    def train(self, queries, responses):
        """Train response generation model"""
        self.logger.info("Training response generation model")
        
        # Prepare data
        data = self.prepare_data(queries, responses)
        
        # Convert tensors to numpy arrays for splitting
        input_ids_np = data['input_ids'].numpy()
        attention_mask_np = data['attention_mask'].numpy()
        decoder_input_ids_np = data['decoder_input_ids'].numpy()
        labels_np = data['labels'].numpy()
        
        # Split data using numpy arrays
        (train_inputs_np, val_inputs_np, 
         train_attention_np, val_attention_np,
         train_decoder_np, val_decoder_np,
         train_labels_np, val_labels_np) = train_test_split(
            input_ids_np,
            attention_mask_np,
            decoder_input_ids_np,
            labels_np,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state']
        )
        
        # Convert back to tensors
        train_inputs = tf.convert_to_tensor(train_inputs_np)
        val_inputs = tf.convert_to_tensor(val_inputs_np)
        train_attention = tf.convert_to_tensor(train_attention_np)
        val_attention = tf.convert_to_tensor(val_attention_np)
        train_decoder = tf.convert_to_tensor(train_decoder_np)
        val_decoder = tf.convert_to_tensor(val_decoder_np)
        train_labels = tf.convert_to_tensor(train_labels_np)
        val_labels = tf.convert_to_tensor(val_labels_np)
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': train_inputs,
                'attention_mask': train_attention,
                'decoder_input_ids': train_decoder
            },
            train_labels
        )).batch(self.config['training']['batch_size'])
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': val_inputs,
                'attention_mask': val_attention,
                'decoder_input_ids': val_decoder
            },
            val_labels
        )).batch(self.config['training']['batch_size'])
        
        # Initialize model
        self.model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(self.config['training']['learning_rate'])
        )
        
        # Train model using custom training loop
        for epoch in range(self.config['training']['epochs']):
            self.logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # Training
            total_loss = 0
            batch_count = 0
            for batch_idx, (inputs, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = self.model(inputs, labels=labels, training=True)
                    loss = outputs.loss
                
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                # ✅ FIXED: Convert numpy array to float
                loss_value = float(loss.numpy())
                total_loss += loss_value
                batch_count += 1
                
                if batch_idx % 2 == 0:
                    # ✅ FIXED: Use the converted float value
                    self.logger.info(f"Batch {batch_idx}, Loss: {loss_value:.4f}")
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            self.logger.info(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        
        self.logger.info("Response generation model training completed")
        return {"loss": avg_loss}
    
    def save_model(self, model_path):
        """Save trained model"""
        if self.model:
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            self.logger.info(f"Model saved to {model_path}")