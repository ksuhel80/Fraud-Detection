import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yaml
import os
import logging

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config_path="config/logging_config.yaml"):
    """Setup logging configuration"""
    with open(config_path, 'r') as f:
        logging_config = yaml.safe_load(f)
    
    logging.config.dictConfig(logging_config)

def ensure_dir(directory):
    """Ensure that a directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(data_path, file_pattern="*.csv"):
    """Load data from CSV files in a directory"""
    data = {}
    for file in os.listdir(data_path):
        if file.endswith('.csv'):
            file_name = file.split('.')[0]
            data[file_name] = pd.read_csv(os.path.join(data_path, file))
    
    return data

def save_data(data, file_path, index=False):
    """Save data to a CSV file"""
    ensure_dir(os.path.dirname(file_path))
    data.to_csv(file_path, index=index)

def plot_transaction_distribution(df, column='transaction_amount', bins=50, title=None):
    """Plot transaction amount distribution"""
    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        title=title or f"Distribution of {column}"
    )
    
    return fig

def plot_time_series(df, date_column='transaction_time', value_column='transaction_amount', 
                    agg_func='count', title=None):
    """Plot time series of transactions"""
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by date
    if agg_func == 'count':
        ts_data = df.groupby(df[date_column].dt.date).size().reset_index(name='count')
        y_label = 'Transaction Count'
    else:
        ts_data = df.groupby(df[date_column].dt.date)[value_column].agg(agg_func).reset_index()
        y_label = f'{agg_func.title()} of {value_column}'
    
    # Create line chart
    fig = px.line(
        ts_data,
        x=date_column,
        y='count' if agg_func == 'count' else value_column,
        title=title or f"{y_label} Over Time"
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=y_label
    )
    
    return fig

def plot_merchant_category_distribution(df, category_column='merchant_category', title=None):
    """Plot merchant category distribution"""
    category_counts = df[category_column].value_counts().reset_index()
    category_counts.columns = [category_column, 'count']
    
    fig = px.bar(
        category_counts,
        x=category_column,
        y='count',
        title=title or "Distribution of Merchant Categories"
    )
    
    return fig

def plot_geographic_distribution(df, lat_col='transaction_location_lat', 
                               lon_col='transaction_location_lon', 
                               color_col=None, title=None):
    """Plot geographic distribution of transactions"""
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        hover_name='transaction_id',
        hover_data=['transaction_amount'],
        zoom=3,
        height=600,
        title=title or "Geographic Distribution of Transactions"
    )
    
    fig.update_layout(mapbox_style="open-street-map")
    
    return fig

def generate_sample_transaction():
    """Generate a sample transaction for testing"""
    return {
        "transaction_id": f"{int(datetime.now().timestamp())}",
        "customer_id": f"{np.random.randint(1, 10000)}",
        "transaction_amount": round(np.random.uniform(10, 1000), 2),
        "transaction_time": (datetime.now() - timedelta(minutes=np.random.randint(1, 1440))).isoformat(),
        "merchant_id": f"{np.random.randint(1, 1000)}",
        "merchant_category": np.random.choice(["retail", "dining", "travel", "entertainment", "gas", "online"]),
        "customer_age": np.random.randint(18, 80),
        "customer_gender": np.random.choice(["M", "F"]),
        "account_tenure": np.random.randint(1, 120),
        "transaction_location_lat": round(np.random.uniform(25, 49), 4),
        "transaction_location_lon": round(np.random.uniform(-125, -66), 4),
        "customer_home_lat": round(np.random.uniform(25, 49), 4),
        "customer_home_lon": round(np.random.uniform(-125, -66), 4)
    }

def calculate_business_impact(metrics, transaction_volume=10000, avg_transaction_value=100):
    """Calculate business impact of model performance"""
    # Extract metrics
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    fpr = 1 - metrics.get('precision', 0)  # Simplified FPR calculation
    
    # Calculate business metrics
    fraud_rate = 0.01  # Assume 1% fraud rate
    
    # Number of fraudulent transactions
    fraud_transactions = transaction_volume * fraud_rate
    
    # Number of detected fraudulent transactions
    detected_fraud = fraud_transactions * recall
    
    # Number of false positives
    false_positives = (transaction_volume - fraud_transactions) * fpr
    
    # Financial impact
    fraud_loss_prevented = detected_fraud * avg_transaction_value
    false_positive_cost = false_positives * 10  # Assume $10 cost per false positive
    
    # Net financial impact
    net_impact = fraud_loss_prevented - false_positive_cost
    
    return {
        'fraud_loss_prevented': fraud_loss_prevented,
        'false_positive_cost': false_positive_cost,
        'net_financial_impact': net_impact,
        'detection_rate': recall,
        'false_positive_rate': fpr
    }