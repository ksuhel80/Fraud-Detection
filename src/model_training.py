import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import joblib
import os
import yaml
import logging
from typing import Dict, List, Tuple, Any
import sklearn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get scikit-learn version
SKLEARN_VERSION = sklearn.__version__.split('.')
SKLEARN_MAJOR = int(SKLEARN_VERSION[0])
SKLEARN_MINOR = int(SKLEARN_VERSION[1])

class FraudDetectionModel:
    def __init__(self, random_state=42, config_path=None):
        if config_path is None:
            config_path = "config/config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_type = None
        self.best_score = None
        self.preprocessor = None
        self.feature_names = None
        self.best_params = {}
        
    def create_model(self, model_type, params=None):
        """Create model based on model_type"""
        if params is None:
            params = {}
            
        if model_type == 'logistic_regression':
            model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                **params
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                **params
            )
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss',
                **params
            )
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                **params
            )
        elif model_type == 'catboost':
            model = CatBoostClassifier(
                random_state=self.random_state,
                verbose=False,
                **params
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return model
    
    def custom_cost_metric(self, y_true, y_pred, false_positive_cost=10, false_negative_cost=100):
        """
        Custom cost-based metric that considers:
        - Cost of false positives (legitimate transactions flagged as fraud)
        - Cost of false negatives (fraudulent transactions missed)
        """
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate total cost
        total_cost = (fp * false_positive_cost) + (fn * false_negative_cost)
        
        # Normalize by number of samples
        normalized_cost = total_cost / len(y_true)
        
        # Return negative cost (since we want to minimize cost, but Optuna maximizes)
        return -normalized_cost
    
    def objective(self, trial, X_train, y_train, cv, model_type, sampling_method='smote'):
        """Objective function for Optuna hyperparameter optimization"""
        # Define hyperparameter search space based on model type
        if model_type == 'logistic_regression':
            params = {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
            }
            # Fix incompatible combinations
            if params['solver'] == 'lbfgs' and params['penalty'] == 'l1':
                raise optuna.TrialPruned()  # Skip this trial
        elif model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 50)
            }
        elif model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 50),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30)
            }
        elif model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 50)
            }
        
        # Create model with current parameters
        model = self.create_model(model_type, params)
        
        # Create pipeline with sampling method
        if sampling_method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
        elif sampling_method == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        else:
            sampler = None
        
        if sampler is not None:
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('model', model)
            ])
        else:
            pipeline = model
        
        # Cross-validation
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Make a copy to avoid modifying the original data
            X_fold_train_processed = X_fold_train.copy()
            X_fold_val_processed = X_fold_val.copy()
            
            # Handle datetime columns - convert to numerical features
            datetime_cols = X_fold_train_processed.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                # Extract datetime features
                X_fold_train_processed[f'{col}_hour'] = X_fold_train_processed[col].dt.hour
                X_fold_train_processed[f'{col}_day_of_week'] = X_fold_train_processed[col].dt.dayofweek
                X_fold_train_processed[f'{col}_day_of_month'] = X_fold_train_processed[col].dt.day
                X_fold_train_processed[f'{col}_month'] = X_fold_train_processed[col].dt.month
                
                X_fold_val_processed[f'{col}_hour'] = X_fold_val_processed[col].dt.hour
                X_fold_val_processed[f'{col}_day_of_week'] = X_fold_val_processed[col].dt.dayofweek
                X_fold_val_processed[f'{col}_day_of_month'] = X_fold_val_processed[col].dt.day
                X_fold_val_processed[f'{col}_month'] = X_fold_val_processed[col].dt.month
                
                # Drop original datetime column
                X_fold_train_processed = X_fold_train_processed.drop(col, axis=1)
                X_fold_val_processed = X_fold_val_processed.drop(col, axis=1)
            
            # Handle categorical columns with one-hot encoding
            categorical_cols = X_fold_train_processed.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                from sklearn.preprocessing import OneHotEncoder
                
                # Use appropriate parameter based on scikit-learn version
                if SKLEARN_MAJOR >= 1 and SKLEARN_MINOR >= 2:
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                else:
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                
                # Fit on training data and transform both training and validation data
                train_encoded = encoder.fit_transform(X_fold_train_processed[categorical_cols])
                val_encoded = encoder.transform(X_fold_val_processed[categorical_cols])
                
                # Create DataFrames with encoded features
                train_encoded_df = pd.DataFrame(train_encoded, index=X_fold_train_processed.index)
                val_encoded_df = pd.DataFrame(val_encoded, index=X_fold_val_processed.index)
                
                # Drop original categorical columns and concatenate encoded features
                X_fold_train_processed = X_fold_train_processed.drop(categorical_cols, axis=1)
                X_fold_val_processed = X_fold_val_processed.drop(categorical_cols, axis=1)
                
                X_fold_train_processed = pd.concat([X_fold_train_processed, train_encoded_df], axis=1)
                X_fold_val_processed = pd.concat([X_fold_val_processed, val_encoded_df], axis=1)
            
            # Convert all column names to strings to avoid mixed types
            X_fold_train_processed.columns = X_fold_train_processed.columns.astype(str)
            X_fold_val_processed.columns = X_fold_val_processed.columns.astype(str)
            
            # Fit pipeline
            pipeline.fit(X_fold_train_processed, y_fold_train)
            
            # Predict probabilities
            y_pred_proba = pipeline.predict_proba(X_fold_val_processed)[:, 1]
            
            # Convert probabilities to binary predictions using optimal threshold
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_fold_val, y_pred, zero_division=0)
            recall = recall_score(y_fold_val, y_pred, zero_division=0)
            f1 = f1_score(y_fold_val, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_fold_val, y_pred_proba)
            pr_auc = average_precision_score(y_fold_val, y_pred_proba)
            cost_metric = self.custom_cost_metric(y_fold_val, y_pred)
            
            # Store F1 score (can be changed to any other metric)
            cv_scores.append(f1)
        
        # Return mean of CV scores
        return np.mean(cv_scores)
    
    def optimize_hyperparameters(self, X_train, y_train, model_types=None, n_trials=50, sampling_method='smote'):
        """Optimize hyperparameters for multiple models using Optuna"""
        if model_types is None:
            model_types = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
        
        # Create stratified time-series cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        
        results = {}
        
        for model_type in model_types:
            logger.info(f"Optimizing hyperparameters for {model_type}...")
            
            # Create Optuna study
            study = optuna.create_study(direction='maximize', study_name=f"{model_type}_optimization")
            
            # Define objective function with fixed data and model type
            def objective(trial):
                return self.objective(trial, X_train, y_train, cv, model_type, sampling_method)
            
            # Optimize
            study.optimize(objective, n_trials=n_trials)
            
            # Store results
            results[model_type] = {
                'best_params': study.best_params,
                'best_score': study.best_value
            }
            
            logger.info(f"Best parameters for {model_type}: {study.best_params}")
            logger.info(f"Best score for {model_type}: {study.best_value}")
        
        # Find the best model type
        best_model_type = max(results, key=lambda k: results[k]['best_score'])
        self.best_model_type = best_model_type
        self.best_score = results[best_model_type]['best_score']
        self.best_params = results[best_model_type]['best_params']
        
        logger.info(f"Best model type: {best_model_type} with score: {self.best_score}")
        
        return results
    
    def preprocess_data(self, X):
        """Preprocess data for model training or prediction"""
        # Make a copy to avoid modifying the original data
        X_processed = X.copy()
        
        # Handle datetime columns - convert to numerical features
        datetime_cols = X_processed.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            # Extract datetime features
            X_processed[f'{col}_hour'] = X_processed[col].dt.hour
            X_processed[f'{col}_day_of_week'] = X_processed[col].dt.dayofweek
            X_processed[f'{col}_day_of_month'] = X_processed[col].dt.day
            X_processed[f'{col}_month'] = X_processed[col].dt.month
            
            # Drop original datetime column
            X_processed = X_processed.drop(col, axis=1)
        
        # Handle categorical columns with one-hot encoding
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            from sklearn.preprocessing import OneHotEncoder
            
            # Use appropriate parameter based on scikit-learn version
            if SKLEARN_MAJOR >= 1 and SKLEARN_MINOR >= 2:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            else:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            
            # Fit and transform data
            encoded = encoder.fit_transform(X_processed[categorical_cols])
            
            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(encoded, index=X_processed.index)
            
            # Drop original categorical columns and concatenate encoded features
            X_processed = X_processed.drop(categorical_cols, axis=1)
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
            
            # Store encoder for later use
            self.encoder = encoder
        
        # Convert all column names to strings to avoid mixed types
        X_processed.columns = X_processed.columns.astype(str)
        
        return X_processed
    
    def train_best_model(self, X_train, y_train, sampling_method='smote'):
        """Train the best model with optimized hyperparameters"""
        if self.best_model_type is None:
            raise ValueError("No model has been optimized yet. Call optimize_hyperparameters first.")
        
        # Preprocess data
        X_train_processed = self.preprocess_data(X_train)
        
        # Create model with best parameters
        model = self.create_model(self.best_model_type, self.best_params)
        
        # Create pipeline with sampling method
        if sampling_method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
        elif sampling_method == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        else:
            sampler = None
        
        if sampler is not None:
            self.best_model = ImbPipeline([
                ('sampler', sampler),
                ('model', model)
            ])
        else:
            self.best_model = model
        
        # Fit pipeline
        self.best_model.fit(X_train_processed, y_train)
        
        # Store feature names if available
        if hasattr(X_train_processed, 'columns'):
            self.feature_names = X_train_processed.columns.tolist()
    
    def predict(self, X):
        """Make binary predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        return self.best_model.predict(X_processed)
    
    def predict_proba(self, X):
        """Predict class probabilities using the best model"""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        return self.best_model.predict_proba(X_processed)
    
    def evaluate(self, X_test, y_test):
        """Evaluate best model performance"""
        # Preprocess data
        X_test_processed = self.preprocess_data(X_test)
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'custom_cost': self.custom_cost_metric(y_test, y_pred)
        }
        
        return metrics
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        
        joblib.dump({
            'pipeline': self.best_model,
            'model_type': self.best_model_type,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'encoder': getattr(self, 'encoder', None)
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['pipeline']
        self.best_model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data['best_params']
        self.best_score = model_data['best_score']
        
        if 'encoder' in model_data:
            self.encoder = model_data['encoder']