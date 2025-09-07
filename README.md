# Credit Card Fraud Detection System

A comprehensive real-time system for detecting fraudulent credit card transactions using machine learning.

## System Overview

This system provides an end-to-end solution for credit card fraud detection, including:

- Data preprocessing and feature engineering
- Multiple machine learning models with hyperparameter optimization
- Real-time transaction scoring API
- Interactive dashboard for monitoring and analysis
- Model retraining capabilities

## Features

### Data Processing & Feature Engineering
- Comprehensive preprocessing pipeline for transaction, customer, and merchant data
- Outlier detection using Isolation Forest and DBSCAN
- Feature engineering including time-based, geographic, and aggregate features

### Advanced Modeling
- Multiple algorithm implementations (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)
- Handling of imbalanced data using SMOTE-NC and ADASYN
- Bayesian hyperparameter optimization with Optuna
- Custom cost-based evaluation metric

### Real-time API
- Flask API with transaction scoring endpoint
- Model retraining capabilities
- System monitoring endpoint
- Comprehensive error handling and logging

### Interactive Dashboard
- Real-time transaction prediction interface
- Fraud probability visualization
- System monitoring metrics
- Historical analysis capabilities

## Project Structure
