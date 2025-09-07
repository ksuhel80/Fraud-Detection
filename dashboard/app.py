import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import time
import yaml
import os

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Load configuration
def load_config():
    try:
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}

config = load_config()

# API endpoints
API_BASE_URL = "http://localhost:5000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
MONITOR_ENDPOINT = f"{API_BASE_URL}/monitor"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Function to get API health status
def get_api_health():
    try:
        response = requests.get(HEALTH_ENDPOINT)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Function to get monitoring data
def get_monitoring_data():
    try:
        response = requests.get(MONITOR_ENDPOINT)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching monitoring data: {str(e)}")
        return None

# Function to make a prediction
def make_prediction(transaction_data):
    try:
        response = requests.post(PREDICT_ENDPOINT, json=transaction_data)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status code: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": str(e)}

# Function to generate sample transaction data
def generate_sample_transaction():
    return {
        "transaction_id": f"{int(time.time())}",
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

# Main application
def main():
    # Header
    st.title("🛡️ Credit Card Fraud Detection Dashboard")
    st.markdown("Real-time monitoring and analysis of credit card transactions")
    
    # Check API health
    health_status = get_api_health()
    
    # Display API status
    if health_status.get("status") == "healthy":
        st.success("API Connection: ✅ Healthy")
    else:
        st.error(f"API Connection: ❌ Unhealthy - {health_status.get('error', 'Unknown error')}")
    
    # Get monitoring data
    monitoring_data = get_monitoring_data()
    
    # Create layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if monitoring_data:
            st.metric("Model Type", monitoring_data.get("model_info", {}).get("model_type", "Unknown"))
    
    with col2:
        if monitoring_data:
            st.metric("Features", monitoring_data.get("model_info", {}).get("feature_count", 0))
    
    with col3:
        if monitoring_data:
            last_updated = monitoring_data.get("model_info", {}).get("last_updated", "Unknown")
            if last_updated != "Unknown":
                last_updated = datetime.fromisoformat(last_updated).strftime("%Y-%m-%d %H:%M:%S")
            st.metric("Last Updated", last_updated)
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Real-time Prediction", "Historical Analysis", "System Monitoring"])
    
    # Tab 1: Real-time Prediction
    with tab1:
        st.header("Real-time Transaction Prediction")
        
        # Input form
        with st.form("prediction_form"):
            st.subheader("Enter Transaction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                transaction_id = st.text_input("Transaction ID", value=f"txn_{int(time.time())}")
                customer_id = st.text_input("Customer ID", value=f"cust_{np.random.randint(1, 10000)}")
                transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0)
                merchant_id = st.text_input("Merchant ID", value=f"merch_{np.random.randint(1, 1000)}")
                merchant_category = st.selectbox("Merchant Category", 
                                                ["retail", "dining", "travel", "entertainment", "gas", "online"])
            
            with col2:
                customer_age = st.slider("Customer Age", min_value=18, max_value=100, value=35)
                customer_gender = st.selectbox("Customer Gender", ["M", "F"])
                account_tenure = st.slider("Account Tenure (months)", min_value=1, max_value=240, value=24)
                transaction_lat = st.number_input("Transaction Latitude", value=40.7128)
                transaction_lon = st.number_input("Transaction Longitude", value=-74.0060)
                home_lat = st.number_input("Home Latitude", value=40.7128)
                home_lon = st.number_input("Home Longitude", value=-74.0060)
            
            submitted = st.form_submit_button("Predict Fraud")
            
            if submitted:
                # Create transaction data
                transaction_data = {
                    "transaction_id": transaction_id,
                    "customer_id": customer_id,
                    "transaction_amount": transaction_amount,
                    "transaction_time": datetime.now().isoformat(),
                    "merchant_id": merchant_id,
                    "merchant_category": merchant_category,
                    "customer_age": customer_age,
                    "customer_gender": customer_gender,
                    "account_tenure": account_tenure,
                    "transaction_location_lat": transaction_lat,
                    "transaction_location_lon": transaction_lon,
                    "customer_home_lat": home_lat,
                    "customer_home_lon": home_lon
                }
                
                # Make prediction
                with st.spinner("Analyzing transaction..."):
                    prediction = make_prediction(transaction_data)
                
                # Display results
                if "error" in prediction:
                    st.error(f"Prediction error: {prediction['error']}")
                else:
                    fraud_probability = prediction["fraud_probability"]
                    is_fraud = prediction["is_fraud"]
                    
                    # Display prediction
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Fraud Probability", f"{fraud_probability:.4f}")
                    
                    with col2:
                        if is_fraud:
                            st.error("⚠️ FRAUD DETECTED")
                        else:
                            st.success("✅ Legitimate Transaction")
                    
                    # Visualize probability
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=fraud_probability,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Fraud Probability"},
                        delta={'reference': 0.5},
                        gauge={
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.25], 'color': "lightgreen"},
                                {'range': [0.25, 0.5], 'color': "yellow"},
                                {'range': [0.5, 0.75], 'color': "orange"},
                                {'range': [0.75, 1], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Sample transaction button
        if st.button("Generate Sample Transaction"):
            sample_transaction = generate_sample_transaction()
            
            # Update form fields with sample data
            st.session_state.transaction_id = sample_transaction["transaction_id"]
            st.session_state.customer_id = sample_transaction["customer_id"]
            st.session_state.transaction_amount = sample_transaction["transaction_amount"]
            st.session_state.merchant_id = sample_transaction["merchant_id"]
            st.session_state.merchant_category = sample_transaction["merchant_category"]
            st.session_state.customer_age = sample_transaction["customer_age"]
            st.session_state.customer_gender = sample_transaction["customer_gender"]
            st.session_state.account_tenure = sample_transaction["account_tenure"]
            st.session_state.transaction_lat = sample_transaction["transaction_location_lat"]
            st.session_state.transaction_lon = sample_transaction["transaction_location_lon"]
            st.session_state.home_lat = sample_transaction["customer_home_lat"]
            st.session_state.home_lon = sample_transaction["customer_home_lon"]
            
            st.rerun()
    
    # Tab 2: Historical Analysis
    with tab2:
        st.header("Historical Fraud Analysis")
        
        # Placeholder for historical data analysis
        st.info("Historical analysis functionality would be implemented here. This would include:")
        st.markdown("""
        - Time-series analysis of fraud patterns
        - Merchant category risk analysis
        - Geographical hotspots for fraud
        - Customer behavior analysis
        - Model performance metrics over time
        """)
        
        # Sample visualization
        st.subheader("Sample Fraud Pattern Visualization")
        
        # Generate sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        fraud_counts = np.random.poisson(lam=5, size=len(dates))
        transaction_counts = np.random.poisson(lam=100, size=len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'fraud_count': fraud_counts,
            'transaction_count': transaction_counts,
            'fraud_rate': fraud_counts / transaction_counts
        })
        
        # Create visualization
        from plotly.subplots import make_subplots
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for transaction counts
        fig.add_trace(
            go.Bar(x=df['date'], y=df['transaction_count'], name="Transaction Count", marker_color='lightblue'),
            secondary_y=False,
        )
        
        # Add line chart for fraud rate
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['fraud_rate'], name="Fraud Rate", marker_color='red'),
            secondary_y=True,
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Date")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Rate", secondary_y=True)
        
        fig.update_layout(title_text="Transaction Volume and Fraud Rate Over Time")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: System Monitoring
    with tab3:
        st.header("System Monitoring")
        
        if monitoring_data:
            # Model information
            st.subheader("Model Information")
            
            model_info = monitoring_data.get("model_info", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
                st.write(f"**Feature Count:** {model_info.get('feature_count', 0)}")
            
            with col2:
                last_updated = model_info.get('last_updated', 'Unknown')
                if last_updated != 'Unknown':
                    last_updated = datetime.fromisoformat(last_updated).strftime("%Y-%m-%d %H:%M:%S")
                st.write(f"**Last Updated:** {last_updated}")
            
            # System statistics
            st.subheader("System Statistics")
            
            system_stats = monitoring_data.get("system_stats", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpu_usage = system_stats.get("cpu_usage", 0)
                st.metric("CPU Usage", f"{cpu_usage}%")
                
                # CPU usage gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=cpu_usage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "CPU Usage (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                memory_usage = system_stats.get("memory_usage", 0)
                st.metric("Memory Usage", f"{memory_usage}%")
                
                # Memory usage gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=memory_usage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Memory Usage (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                disk_usage = system_stats.get("disk_usage", 0)
                st.metric("Disk Usage", f"{disk_usage}%")
                
                # Disk usage gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=disk_usage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Disk Usage (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Refresh button
            if st.button("Refresh Monitoring Data"):
                st.rerun()
        else:
            st.error("Unable to fetch monitoring data")

# Run the application
if __name__ == "__main__":
    main()