import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yaml
from src.data_preprocessing import DataPreprocessor
from src.model_training import FraudDetectionModel
from src.evaluation import evaluate_model
import joblib
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Load data
    logger.info("Loading data...")
    transactions = pd.read_csv(os.path.join(config["data"]["raw_path"], "transactions.csv"))
    customers = pd.read_csv(os.path.join(config["data"]["raw_path"], "customers.csv"))
    merchants = pd.read_csv(os.path.join(config["data"]["raw_path"], "merchants.csv"))
    fraud_labels = pd.read_csv(os.path.join(config["data"]["raw_path"], "fraud_labels.csv"))
    
    # Merge data
    data = transactions.merge(customers, on="customer_id")
    data = data.merge(merchants, on="merchant_id")
    data = data.merge(fraud_labels, on="transaction_id")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess data
    logger.info("Preprocessing data...")
    processed_data = preprocessor.preprocess(data)
    
    # Split features and target
    X = processed_data.drop("is_fraud", axis=1)
    y = processed_data["is_fraud"]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["training"]["test_size"], 
        random_state=config["model"]["random_state"],
        stratify=y
    )
    
    # Initialize model
    model = FraudDetectionModel(random_state=config["model"]["random_state"])
    
    # Define model types to optimize
    model_types = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
    
    # Optimize hyperparameters for all models
    logger.info("Optimizing hyperparameters for all models...")
    optimization_results = model.optimize_hyperparameters(
        X_train, y_train,
        model_types=model_types,
        n_trials=30,  # Reduced for faster execution
        sampling_method=config["training"]["sampling_method"]
    )
    
    # Log optimization results
    for model_type, results in optimization_results.items():
        logger.info(f"{model_type}: Best Score = {results['best_score']:.4f}, Best Params = {results['best_params']}")
    
    # Train the best model
    logger.info(f"Training the best model: {model.best_model_type}")
    model.train_best_model(
        X_train, y_train,
        sampling_method=config["training"]["sampling_method"]
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Test metrics: {metrics}")
    
    # Save model
    os.makedirs(config["model"]["save_path"], exist_ok=True)
    model.save_model(os.path.join(config["model"]["save_path"], "fraud_detection_model.pkl"))
    
    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(config["model"]["save_path"], "preprocessor.pkl"))
    
    # Save optimization results
    os.makedirs(os.path.join(config["model"]["save_path"], "optimization_results"), exist_ok=True)
    with open(os.path.join(config["model"]["save_path"], "optimization_results", "all_models_results.json"), "w") as f:
        json.dump(optimization_results, f, indent=4)
    
    # Save best model info
    best_model_info = {
        "model_type": model.best_model_type,
        "best_score": model.best_score,
        "best_params": model.best_params,
        "test_metrics": metrics
    }
    
    with open(os.path.join(config["model"]["save_path"], "optimization_results", "best_model_info.json"), "w") as f:
        json.dump(best_model_info, f, indent=4)
    
    logger.info("Model training completed successfully!")
    logger.info(f"Best model: {model.best_model_type} with validation score: {model.best_score:.4f}")
    logger.info(f"Test metrics: {metrics}")

if __name__ == "__main__":
    main()