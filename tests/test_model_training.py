import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from src.model_training import FraudDetectionModel

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })
    
    # Create imbalanced target (10% fraud)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y[fraud_indices] = 1
    
    return X, y

def test_create_model():
    # Test different model types
    model_types = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
    
    for model_type in model_types:
        model = FraudDetectionModel(model_type=model_type)
        model_instance = model.create_model()
        
        assert model_instance is not None

def test_optimize_hyperparameters(sample_data):
    X, y = sample_data
    model = FraudDetectionModel(model_type='logistic_regression')
    
    # Test with a small number of trials for speed
    best_params, best_score = model.optimize_hyperparameters(X, y, n_trials=2)
    
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert len(best_params) > 0

def test_train_predict(sample_data):
    X, y = sample_data
    model = FraudDetectionModel(model_type='logistic_regression')
    
    # Train model
    model.train(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Check predictions
    assert len(y_pred) == len(y)
    assert len(y_pred_proba) == len(y)
    assert y_pred_proba.shape[1] == 2  # Binary classification
    assert set(y_pred).issubset({0, 1})

def test_evaluate(sample_data):
    X, y = sample_data
    model = FraudDetectionModel(model_type='logistic_regression')
    
    # Train model
    model.train(X, y)
    
    # Evaluate model
    metrics = model.evaluate(X, y)
    
    # Check metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert 'pr_auc' in metrics
    assert 'custom_cost' in metrics
    
    # Check metric values
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1
    assert 0 <= metrics['pr_auc'] <= 1
    assert metrics['custom_cost'] <= 0  # Negative cost

def test_save_load_model(sample_data, tmp_path):
    X, y = sample_data
    model = FraudDetectionModel(model_type='logistic_regression')
    
    # Train model
    model.train(X, y)
    
    # Save model
    model_path = tmp_path / "test_model.pkl"
    model.save_model(str(model_path))
    
    # Load model
    new_model = FraudDetectionModel(model_type='logistic_regression')
    new_model.load_model(str(model_path))
    
    # Make predictions with loaded model
    y_pred = new_model.predict(X)
    y_pred_proba = new_model.predict_proba(X)
    
    # Check predictions
    assert len(y_pred) == len(y)
    assert len(y_pred_proba) == len(y)