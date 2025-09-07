import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import os
import yaml
import psutil

app = Flask(__name__)

# Configure logging
with open("config/logging_config.yaml", "r") as f:
    logging_config = yaml.safe_load(f)
    logging.config.dictConfig(logging_config)

logger = logging.getLogger(__name__)

# Global variables to store models and preprocessors
model = None
preprocessor = None
model_metadata = {}

# Load model and preprocessor at startup
def load_model_and_preprocessor():
    global model, preprocessor, model_metadata
    
    try:
        # Paths to model and preprocessor files
        model_path = os.environ.get('MODEL_PATH', 'models/fraud_detection_model.pkl')
        preprocessor_path = os.environ.get('PREPROCESSOR_PATH', 'models/preprocessor.pkl')
        
        # Load model
        model_data = joblib.load(model_path)
        model = model_data['pipeline']
        model_metadata['model_type'] = model_data['model_type']
        model_metadata['feature_names'] = model_data['feature_names']
        model_metadata['best_params'] = model_data['best_params']
        
        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)
        
        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model and preprocessor: {str(e)}")
        raise

# Initialize model and preprocessor
load_model_and_preprocessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_type': model_metadata.get('model_type', 'unknown')
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Transaction scoring endpoint"""
    try:
        # Get transaction data from request
        transaction_data = request.json
        
        if not transaction_data:
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Validate required fields
        required_fields = model_metadata.get('feature_names', [])
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Preprocess data
        processed_data = preprocessor.transform(df)
        
        # Make prediction
        fraud_probability = model.predict_proba(processed_data)[0, 1]
        is_fraud = int(fraud_probability > 0.5)  # Using 0.5 as threshold, can be adjusted
        
        # Log prediction
        logger.info(f"Transaction prediction: fraud_probability={fraud_probability:.4f}, is_fraud={is_fraud}")
        
        # Return prediction
        return jsonify({
            'transaction_id': transaction_data.get('transaction_id', 'unknown'),
            'fraud_probability': float(fraud_probability),
            'is_fraud': is_fraud,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Model retraining endpoint"""
    try:
        # Get training data from request
        training_data = request.json
        
        if not training_data or 'transactions' not in training_data:
            return jsonify({'error': 'No training data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data['transactions'])
        
        # Validate required fields
        if 'is_fraud' not in df.columns:
            return jsonify({'error': 'Training data must include "is_fraud" labels'}), 400
        
        # Split features and target
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        # Preprocess data
        processed_X = preprocessor.transform(X)
        
        # Retrain model
        model.fit(processed_X, y)
        
        # Evaluate model on training data
        y_pred = model.predict(processed_X)
        y_pred_proba = model.predict_proba(processed_X)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        
        metrics = {
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'pr_auc': average_precision_score(y, y_pred_proba)
        }
        
        # Save updated model
        model_path = os.environ.get('MODEL_PATH', 'models/fraud_detection_model.pkl')
        joblib.dump({
            'pipeline': model,
            'model_type': model_metadata['model_type'],
            'feature_names': model_metadata['feature_names'],
            'best_params': model_metadata['best_params'],
            'retrain_timestamp': datetime.now().isoformat()
        }, model_path)
        
        # Log retraining
        logger.info(f"Model retrained with {len(df)} samples. New metrics: {metrics}")
        
        # Return retraining results
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'training_samples': len(df),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error during model retraining: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/monitor', methods=['GET'])
def monitor():
    """Monitoring endpoint"""
    try:
        # Get model metadata
        model_info = {
            'model_type': model_metadata.get('model_type', 'unknown'),
            'feature_count': len(model_metadata.get('feature_names', [])),
            'last_updated': model_metadata.get('retrain_timestamp', 'unknown')
        }
        
        # Get basic system stats
        system_stats = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        # Return monitoring information
        return jsonify({
            'model_info': model_info,
            'system_stats': system_stats,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error during monitoring: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load configuration
    config = {}
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
    
    # Run the app
    app.run(
        host=config.get('api', {}).get('host', '0.0.0.0'),
        port=config.get('api', {}).get('port', 5000),
        debug=True
    )