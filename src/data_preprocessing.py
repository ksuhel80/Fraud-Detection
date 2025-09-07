import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from datetime import datetime
import joblib
import os
import yaml
import logging.config  

class DataPreprocessor:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = "config/config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.numeric_features = ['transaction_amount', 'customer_age', 'account_tenure']
        self.categorical_features = ['merchant_category', 'customer_gender']
        self.datetime_features = ['transaction_time']
        self.geo_features = ['transaction_location_lat', 'transaction_location_lon', 
                            'customer_home_lat', 'customer_home_lon']
        
        self.preprocessor = self._create_preprocessor()
        self.outlier_detector = IsolationForest(contamination=0.01, random_state=42)
        
    def _create_preprocessor(self):
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor
    
    def extract_datetime_features(self, df):
        """Extract features from datetime columns"""
        df_copy = df.copy()
        
        for col in self.datetime_features:
            if col in df_copy.columns:
                df_copy[col] = pd.to_datetime(df_copy[col])
                df_copy[f'{col}_hour'] = df_copy[col].dt.hour
                df_copy[f'{col}_day_of_week'] = df_copy[col].dt.dayofweek
                df_copy[f'{col}_day_of_month'] = df_copy[col].dt.day
                df_copy[f'{col}_month'] = df_copy[col].dt.month
        
        return df_copy
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two geographic coordinates"""
        # Haversine formula implementation
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Earth radius in km
        return c * r
    
    def extract_geo_features(self, df):
        """Extract features from geographic data"""
        df_copy = df.copy()
        
        if all(col in df_copy.columns for col in self.geo_features):
            # Calculate distance between transaction location and customer's home
            df_copy['distance_from_home'] = self.calculate_distance(
                df_copy['transaction_location_lat'], 
                df_copy['transaction_location_lon'],
                df_copy['customer_home_lat'], 
                df_copy['customer_home_lon']
            )
            
            # Binary feature for whether transaction is far from home
            df_copy['is_far_from_home'] = (df_copy['distance_from_home'] > 100).astype(int)
        
        return df_copy
    
    def detect_outliers(self, df):
        """Detect outliers using Isolation Forest and DBSCAN"""
        df_copy = df.copy()
        
        # Select numeric features for outlier detection
        numeric_data = df_copy[self.numeric_features].fillna(0)
        
        # Isolation Forest
        isolation_outliers = self.outlier_detector.fit_predict(numeric_data)
        df_copy['isolation_outlier'] = (isolation_outliers == -1).astype(int)
        
        # DBSCAN
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        dbscan = DBSCAN(eps=3.0, min_samples=10)
        dbscan_clusters = dbscan.fit_predict(scaled_data)
        df_copy['dbscan_outlier'] = (dbscan_clusters == -1).astype(int)
        
        # Combined outlier flag
        df_copy['is_outlier'] = ((df_copy['isolation_outlier'] == 1) | 
                                 (df_copy['dbscan_outlier'] == 1)).astype(int)
        
        return df_copy
    
    def create_aggregate_features(self, df):
        """Create aggregate features based on historical patterns"""
        df_copy = df.copy()
        
        # Customer-level aggregations
        customer_agg = df_copy.groupby('customer_id').agg({
            'transaction_amount': ['mean', 'std', 'max', 'min'],
            'transaction_id': 'count'
        })
        
        # Flatten column names
        customer_agg.columns = ['_'.join(col).strip() for col in customer_agg.columns.values]
        customer_agg.reset_index(inplace=True)
        
        # Merge back to main dataframe
        df_copy = pd.merge(df_copy, customer_agg, on='customer_id', how='left')
        
        # Merchant-level aggregations
        merchant_agg = df_copy.groupby('merchant_id').agg({
            'transaction_amount': ['mean', 'std'],
            'is_fraud': 'mean'  # Fraud rate by merchant
        })
        
        # Flatten column names
        merchant_agg.columns = ['_'.join(col).strip() for col in merchant_agg.columns.values]
        merchant_agg.reset_index(inplace=True)
        
        # Rename fraud rate column
        merchant_agg = merchant_agg.rename(columns={'is_fraud_mean': 'merchant_fraud_rate'})
        
        # Merge back to main dataframe
        df_copy = pd.merge(df_copy, merchant_agg, on='merchant_id', how='left')
        
        return df_copy
    
    def preprocess(self, df):
        """Complete preprocessing pipeline"""
        # Extract datetime features
        df = self.extract_datetime_features(df)
        
        # Extract geographic features
        df = self.extract_geo_features(df)
        
        # Create aggregate features
        df = self.create_aggregate_features(df)
        
        # Detect outliers
        df = self.detect_outliers(df)
        
        return df
    
    def transform(self, df):
        """Transform data using the preprocessor"""
        # First apply the preprocessing steps
        df = self.preprocess(df)
        
        # Convert all column names to strings to avoid mixed types
        df.columns = df.columns.astype(str)
        
        # Then apply the column transformations
        return self.preprocessor.transform(df)
    
    def save(self, filepath):
        """Save the preprocessor"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a saved preprocessor"""
        return joblib.load(filepath)