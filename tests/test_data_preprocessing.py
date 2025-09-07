import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'transaction_id': ['1', '2', '3'],
        'customer_id': ['1', '2', '3'],
        'merchant_id': ['1', '2', '3'],
        'transaction_amount': [100.0, 200.0, 300.0],
        'transaction_time': ['2023-01-01 08:00:00', '2023-01-01 09:00:00', '2023-01-01 10:00:00'],
        'transaction_location_lat': [40.7128, 34.0522, 41.8781],
        'transaction_location_lon': [-74.0060, -118.2437, -87.6298],
        'customer_age': [35, 42, 28],
        'customer_gender': ['M', 'F', 'M'],
        'account_tenure': [24, 36, 12],
        'customer_home_lat': [40.7589, 34.0195, 41.8819],
        'customer_home_lon': [-73.9851, -118.4912, -87.6278],
        'merchant_category': ['retail', 'dining', 'travel'],
        'is_fraud': [0, 0, 1]
    })

def test_extract_datetime_features(sample_data):
    preprocessor = DataPreprocessor()
    result = preprocessor.extract_datetime_features(sample_data)
    
    assert 'transaction_time_hour' in result.columns
    assert 'transaction_time_day_of_week' in result.columns
    assert 'transaction_time_day_of_month' in result.columns
    assert 'transaction_time_month' in result.columns
    
    # Check specific values
    assert result['transaction_time_hour'].iloc[0] == 8
    assert result['transaction_time_day_of_week'].iloc[0] == 6  # Sunday

def test_extract_geo_features(sample_data):
    preprocessor = DataPreprocessor()
    result = preprocessor.extract_geo_features(sample_data)
    
    assert 'distance_from_home' in result.columns
    assert 'is_far_from_home' in result.columns
    
    # Check distance calculation
    assert result['distance_from_home'].iloc[0] > 0
    assert result['is_far_from_home'].iloc[0] == 0  # Should be close to home

def test_detect_outliers(sample_data):
    preprocessor = DataPreprocessor()
    result = preprocessor.detect_outliers(sample_data)
    
    assert 'isolation_outlier' in result.columns
    assert 'dbscan_outlier' in result.columns
    assert 'is_outlier' in result.columns
    
    # Check that outlier flags are binary
    assert set(result['isolation_outlier'].unique()).issubset({0, 1})
    assert set(result['dbscan_outlier'].unique()).issubset({0, 1})
    assert set(result['is_outlier'].unique()).issubset({0, 1})

def test_create_aggregate_features(sample_data):
    preprocessor = DataPreprocessor()
    result = preprocessor.create_aggregate_features(sample_data)
    
    # Check for aggregate features
    assert 'transaction_amount_mean' in result.columns
    assert 'transaction_amount_std' in result.columns
    assert 'transaction_amount_max' in result.columns
    assert 'transaction_amount_min' in result.columns
    assert 'transaction_id_count' in result.columns
    assert 'merchant_fraud_rate' in result.columns

def test_preprocess(sample_data):
    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess(sample_data)
    
    # Check that all preprocessing steps are applied
    assert 'transaction_time_hour' in result.columns
    assert 'distance_from_home' in result.columns
    assert 'is_outlier' in result.columns
    assert 'transaction_amount_mean' in result.columns
    assert 'merchant_fraud_rate' in result.columns

def test_transform(sample_data):
    preprocessor = DataPreprocessor()
    result = preprocessor.transform(sample_data)
    
    # Check that transform returns a numpy array
    assert isinstance(result, np.ndarray)
    
    # Check shape
    assert result.shape[0] == len(sample_data)