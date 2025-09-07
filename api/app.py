import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
import logging.config
from datetime import datetime
import yaml
import psutil
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import sklearn

app = Flask(__name__)

# Configure logging
with open("config/logging_config.yaml", "r") as f:
    logging_config = yaml.safe_load(f)
    logging.config.dictConfig(logging_config)

logger = logging.getLogger(__name__)

# Get scikit-learn version
SKLEARN_VERSION = sklearn.__version__.split('.')
SKLEARN_MAJOR = int(SKLEARN_VERSION[0])
SKLEARN_MINOR = int(SKLEARN_VERSION[1])

# Global variables to store models and preprocessors
model = None
model_metadata = {}
feature_names = None

# Load model at startup
def load_model():
    global model, model_metadata, feature_names
    
    try:
        # Path to model file
        model_path = os.environ.get('MODEL_PATH', 'models/fraud_detection_model.pkl')
        
        # Load model
        model_data = joblib.load(model_path)
        model = model_data['pipeline']
        model_metadata['model_type'] = model_data['model_type']
        model_metadata['feature_names'] = model_data['feature_names']
        model_metadata['best_params'] = model_data['best_params']
        feature_names = model_data['feature_names']
        
        logger.info("Model loaded successfully")
        logger.info(f"Model expects {len(feature_names)} features")
        logger.info(f"Feature names: {feature_names[:10]}...")  # Log first 10 feature names
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Initialize model
load_model()

def create_preprocessor():
    """Create a new preprocessor for prediction"""
    try:
        # Define numeric and categorical features
        numeric_features = ['transaction_amount', 'customer_age', 'account_tenure']
        categorical_features = ['merchant_category', 'customer_gender']
        
        # Create transformers
        numeric_transformer = StandardScaler()
        
        # Use appropriate parameter based on scikit-learn version
        if SKLEARN_MAJOR >= 1 and SKLEARN_MINOR >= 2:
            categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        else:
            categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Fit the preprocessor with sample data
        # Create sample data that covers all possible categories
        sample_data = pd.DataFrame({
            'transaction_amount': [100.0, 200.0, 300.0],
            'customer_age': [25, 35, 45],
            'account_tenure': [12, 24, 36],
            'merchant_category': ['retail', 'dining', 'travel'],
            'customer_gender': ['M', 'F', 'M']
        })
        
        # Fit the preprocessor
        preprocessor.fit(sample_data)
        
        return preprocessor
    except Exception as e:
        logger.error(f"Error creating preprocessor: {str(e)}")
        raise

def preprocess_transaction_data(df):
    """Preprocess transaction data for prediction"""
    try:
        # Make a copy to avoid modifying the original data
        df_processed = df.copy()
        
        # Apply preprocessing steps manually to create all features expected by the model
        # Step 1: Extract datetime features
        datetime_cols = df_processed.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col])
                df_processed[f'{col}_hour'] = df_processed[col].dt.hour
                df_processed[f'{col}_day_of_week'] = df_processed[col].dt.dayofweek
                df_processed[f'{col}_day_of_month'] = df_processed[col].dt.day
                df_processed[f'{col}_month'] = df_processed[col].dt.month
        
        # Step 2: Extract geographic features
        geo_features = ['transaction_location_lat', 'transaction_location_lon', 
                        'customer_home_lat', 'customer_home_lon']
        if all(col in df_processed.columns for col in geo_features):
            # Calculate distance between transaction location and customer's home
            lat1 = df_processed['transaction_location_lat'].values
            lon1 = df_processed['transaction_location_lon'].values
            lat2 = df_processed['customer_home_lat'].values
            lon2 = df_processed['customer_home_lon'].values
            
            # Haversine formula implementation
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # Earth radius in km
            df_processed['distance_from_home'] = c * r
            
            # Binary feature for whether transaction is far from home
            df_processed['is_far_from_home'] = (df_processed['distance_from_home'] > 100).astype(int)
        
        # Step 3: Detect outliers
        numeric_features = ['transaction_amount', 'customer_age', 'account_tenure']
        numeric_data = df_processed[numeric_features].fillna(0)
        
        # Isolation Forest
        from sklearn.ensemble import IsolationForest
        outlier_detector = IsolationForest(contamination=0.01, random_state=42)
        isolation_outliers = outlier_detector.fit_predict(numeric_data)
        df_processed['isolation_outlier'] = (isolation_outliers == -1).astype(int)
        
        # DBSCAN
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        dbscan = DBSCAN(eps=3.0, min_samples=10)
        dbscan_clusters = dbscan.fit_predict(scaled_data)
        df_processed['dbscan_outlier'] = (dbscan_clusters == -1).astype(int)
        
        # Combined outlier flag
        df_processed['is_outlier'] = ((df_processed['isolation_outlier'] == 1) | 
                                      (df_processed['dbscan_outlier'] == 1)).astype(int)
        
        # Step 4: Create aggregate features (using default values since we don't have historical data)
        # Customer-level aggregate features
        df_processed['transaction_amount_mean'] = df_processed['transaction_amount']  # Default to current amount
        df_processed['transaction_amount_std'] = 0.0  # Default to 0
        df_processed['transaction_amount_max'] = df_processed['transaction_amount']  # Default to current amount
        df_processed['transaction_amount_min'] = df_processed['transaction_amount']  # Default to current amount
        df_processed['transaction_id_count'] = 1  # Default to 1 (single transaction)
        
        # Merchant-level aggregate features
        df_processed['transaction_amount_mean'] = df_processed['transaction_amount']  # Default to current amount
        df_processed['transaction_amount_std'] = 0.0  # Default to 0
        df_processed['merchant_fraud_rate'] = 0.01  # Default to low fraud rate
        
        # Convert all column names to strings to avoid mixed types
        df_processed.columns = df_processed.columns.astype(str)
        
        # Create a new preprocessor and transform the data
        preprocessor = create_preprocessor()
        processed_data = preprocessor.transform(df_processed)
        
        # Now we need to ensure we have exactly 141 features in the right order
        # Create a DataFrame with the processed data
        processed_df = pd.DataFrame(processed_data)
        
        # If we don't have enough features, add missing ones with default values
        if processed_df.shape[1] < 141:
            logger.warning(f"Only {processed_df.shape[1]} features after preprocessing, adding {141 - processed_df.shape[1]} missing features")
            
            # Add missing features with default values (0)
            for i in range(processed_df.shape[1], 141):
                processed_df[i] = 0.0
        
        # If we have too many features, truncate to 141
        if processed_df.shape[1] > 141:
            logger.warning(f"Too many features ({processed_df.shape[1]}) after preprocessing, truncating to 141")
            processed_df = processed_df.iloc[:, :141]
        
        # Ensure the columns are in the right order (if we have feature names)
        if feature_names is not None and len(feature_names) == 141:
            # Create a DataFrame with all feature names and default values
            final_df = pd.DataFrame(0.0, index=processed_df.index, columns=feature_names)
            
            # Fill in the values we have
            for col in processed_df.columns:
                if col in final_df.columns:
                    final_df[col] = processed_df[col]
            
            processed_data = final_df.values
        else:
            # If we don't have feature names, just use the first 141 columns
            processed_data = processed_df.iloc[:, :141].values
        
        logger.info(f"Final data shape: {processed_data.shape}")
        
        return processed_data
    except Exception as e:
        logger.error(f"Error in preprocess_transaction_data: {str(e)}")
        raise

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
            return jsonify({
                'error': 'No transaction data provided',
                'details': 'The request body is empty'
            }), 400
        
        # Log the incoming data for debugging
        logger.info(f"Received transaction data: {transaction_data}")
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame([transaction_data])
        except Exception as e:
            return jsonify({
                'error': 'Failed to create DataFrame from transaction data',
                'details': str(e)
            }), 400
        
        # Log the columns we received
        logger.info(f"Received columns: {list(df.columns)}")
        
        # Preprocess data
        try:
            processed_data = preprocess_transaction_data(df)
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            return jsonify({
                'error': 'Data preprocessing failed',
                'details': str(e)
            }), 400
        
        # Log the shape after preprocessing
        logger.info(f"Data shape after preprocessing: {processed_data.shape}")
        
        # Make prediction
        try:
            fraud_probability = model.predict_proba(processed_data)[0, 1]
            is_fraud = int(fraud_probability > 0.5)  # Using 0.5 as threshold, can be adjusted
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({
                'error': 'Prediction failed',
                'details': str(e)
            }), 500
        
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
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Model retraining endpoint"""
    try:
        # Get training data from request
        training_data = request.json
        
        if not training_data or 'transactions' not in training_data:
            return jsonify({
                'error': 'No training data provided',
                'details': 'The request must contain a "transactions" field'
            }), 400
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(training_data['transactions'])
        except Exception as e:
            return jsonify({
                'error': 'Failed to create DataFrame from training data',
                'details': str(e)
            }), 400
        
        # Validate required fields
        if 'is_fraud' not in df.columns:
            return jsonify({
                'error': 'Missing required field',
                'details': 'Training data must include "is_fraud" labels'
            }), 400
        
        # Split features and target
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        # Preprocess data
        try:
            # Create a new preprocessor for training
            preprocessor = create_preprocessor()
            
            # Apply preprocessing steps manually
            X_processed = X.copy()
            
            # Extract datetime features
            datetime_cols = X_processed.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                if col in X_processed.columns:
                    X_processed[col] = pd.to_datetime(X_processed[col])
                    X_processed[f'{col}_hour'] = X_processed[col].dt.hour
                    X_processed[f'{col}_day_of_week'] = X_processed[col].dt.dayofweek
                    X_processed[f'{col}_day_of_month'] = X_processed[col].dt.day
                    X_processed[f'{col}_month'] = X_processed[col].dt.month
            
            # Extract geographic features
            geo_features = ['transaction_location_lat', 'transaction_location_lon', 
                            'customer_home_lat', 'customer_home_lon']
            if all(col in X_processed.columns for col in geo_features):
                # Calculate distance between transaction location and customer's home
                lat1 = X_processed['transaction_location_lat'].values
                lon1 = X_processed['transaction_location_lon'].values
                lat2 = X_processed['customer_home_lat'].values
                lon2 = X_processed['customer_home_lon'].values
                
                # Haversine formula implementation
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                r = 6371  # Earth radius in km
                X_processed['distance_from_home'] = c * r
                
                # Binary feature for whether transaction is far from home
                X_processed['is_far_from_home'] = (X_processed['distance_from_home'] > 100).astype(int)
            
            # Detect outliers
            numeric_features = ['transaction_amount', 'customer_age', 'account_tenure']
            numeric_data = X_processed[numeric_features].fillna(0)
            
            # Isolation Forest
            from sklearn.ensemble import IsolationForest
            outlier_detector = IsolationForest(contamination=0.01, random_state=42)
            isolation_outliers = outlier_detector.fit_predict(numeric_data)
            X_processed['isolation_outlier'] = (isolation_outliers == -1).astype(int)
            
            # DBSCAN
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            dbscan = DBSCAN(eps=3.0, min_samples=10)
            dbscan_clusters = dbscan.fit_predict(scaled_data)
            X_processed['dbscan_outlier'] = (dbscan_clusters == -1).astype(int)
            
            # Combined outlier flag
            X_processed['is_outlier'] = ((X_processed['isolation_outlier'] == 1) | 
                                          (X_processed['dbscan_outlier'] == 1)).astype(int)
            
            # Create aggregate features
            # Customer-level aggregations
            customer_agg = X_processed.groupby('customer_id').agg({
                'transaction_amount': ['mean', 'std', 'max', 'min'],
                'transaction_id': 'count'
            })
            
            # Flatten column names
            customer_agg.columns = ['_'.join(col).strip() for col in customer_agg.columns.values]
            customer_agg.reset_index(inplace=True)
            
            # Merge back to main dataframe
            X_processed = pd.merge(X_processed, customer_agg, on='customer_id', how='left')
            
            # Merchant-level aggregations
            merchant_agg = X_processed.groupby('merchant_id').agg({
                'transaction_amount': ['mean', 'std'],
                'is_fraud': 'mean'  # Fraud rate by merchant
            })
            
            # Flatten column names
            merchant_agg.columns = ['_'.join(col).strip() for col in merchant_agg.columns.values]
            merchant_agg.reset_index(inplace=True)
            
            # Rename fraud rate column
            merchant_agg = merchant_agg.rename(columns={'is_fraud_mean': 'merchant_fraud_rate'})
            
            # Merge back to main dataframe
            X_processed = pd.merge(X_processed, merchant_agg, on='merchant_id', how='left')
            
            # Convert all column names to strings to avoid mixed types
            X_processed.columns = X_processed.columns.astype(str)
            
            # Apply the column transformations (scaling, encoding)
            processed_X = preprocessor.transform(X_processed)
        except Exception as e:
            return jsonify({
                'error': 'Data preprocessing failed',
                'details': str(e)
            }), 400
        
        # Retrain model
        try:
            model.fit(processed_X, y)
        except Exception as e:
            return jsonify({
                'error': 'Model training failed',
                'details': str(e)
            }), 500
        
        # Evaluate model on training data
        try:
            y_pred = model.predict(processed_X)
            y_pred_proba = model.predict_proba(processed_X)[:, 1]
        except Exception as e:
            return jsonify({
                'error': 'Model evaluation failed',
                'details': str(e)
            }), 500
        
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
        try:
            joblib.dump({
                'pipeline': model,
                'model_type': model_metadata['model_type'],
                'feature_names': model_metadata['feature_names'],
                'best_params': model_metadata['best_params'],
                'retrain_timestamp': datetime.now().isoformat()
            }, model_path)
        except Exception as e:
            return jsonify({
                'error': 'Failed to save updated model',
                'details': str(e)
            }), 500
        
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
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

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
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

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