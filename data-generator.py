import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize faker
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_customers(n_customers=500):
    """Generate customer demographic data"""
    customers = []
    for i in range(1, n_customers + 1):
        customer = {
            'customer_id': i,
            'customer_age': np.random.normal(45, 15),
            'customer_gender': random.choice(['M', 'F']),
            'account_tenure': np.random.exponential(500),
            'customer_home_lat': fake.latitude(),
            'customer_home_lon': fake.longitude()
        }
        # Ensure age is within reasonable bounds
        customer['customer_age'] = max(18, min(100, int(customer['customer_age'])))
        # Ensure tenure is at least 1 day
        customer['account_tenure'] = max(1, int(customer['account_tenure']))
        customers.append(customer)
    
    return pd.DataFrame(customers)

def generate_merchants(n_merchants=100):
    """Generate merchant category information"""
    categories = ['Retail', 'Food & Dining', 'Travel', 'Entertainment', 
                 'Services', 'Healthcare', 'Online Retail', 'Gas/Energy',
                 'Groceries', 'Electronics', 'Clothing', 'Other']
    
    merchants = []
    for i in range(1, n_merchants + 1):
        merchant = {
            'merchant_id': i,
            'merchant_category': random.choice(categories),
            'merchant_name': fake.company()
        }
        merchants.append(merchant)
    
    return pd.DataFrame(merchants)

def generate_geolocation_reference(n_locations=200):
    """Generate geolocation reference data"""
    locations = []
    for _ in range(n_locations):
        location = {
            'location_lat': fake.latitude(),
            'location_lon': fake.longitude(),
            'country': fake.country(),
            'city': fake.city(),
            'region': fake.state()
        }
        locations.append(location)
    
    return pd.DataFrame(locations)

def generate_transactions_and_fraud_labels(n_transactions=10000, n_customers=500, n_merchants=100):
    """Generate transaction records with fraud patterns and labels"""
    transactions = []
    fraud_labels = []
    start_date = datetime(2023, 1, 1)
    
    # Define fraud patterns
    high_risk_hours = [0, 1, 2, 3, 22, 23]  # Late night hours
    high_risk_categories = ['Travel', 'Entertainment', 'Online Retail', 'Electronics']
    high_risk_amount_threshold = 500  # Transactions above this are more likely to be fraudulent
    
    # Generate some known fraudulent customers and merchants
    fraudulent_customers = random.sample(range(1, n_customers + 1), 10)
    fraudulent_merchants = random.sample(range(1, n_merchants + 1), 5)
    
    for i in range(1, n_transactions + 1):
        customer_id = random.randint(1, n_customers)
        merchant_id = random.randint(1, n_merchants)
        
        # Generate transaction time (more transactions during day time)
        hour = np.random.normal(14, 4)
        hour = int(max(0, min(23, hour)))
        time_delta = timedelta(days=random.randint(0, 365), hours=hour, minutes=random.randint(0, 59))
        transaction_time = start_date + time_delta
        
        # Generate amount (right-skewed distribution)
        amount = np.random.exponential(100)
        
        # Base fraud probability
        fraud_prob = 0.02
        
        # Increase fraud probability based on patterns
        if hour in high_risk_hours:
            fraud_prob *= 3
        if amount > high_risk_amount_threshold:
            fraud_prob *= 2
        if customer_id in fraudulent_customers:
            fraud_prob *= 5
        if merchant_id in fraudulent_merchants:
            fraud_prob *= 4
        
        # Randomly decide if this transaction is fraudulent
        is_fraud = np.random.random() < fraud_prob
        
        # Generate transaction location (for legitimate transactions, use customer's home region)
        if is_fraud and np.random.random() < 0.7:
            # For fraudulent transactions, often far from home
            transaction_lat = fake.latitude()
            transaction_lon = fake.longitude()
        else:
            # For legitimate transactions, usually near home
            transaction_lat = fake.latitude()
            transaction_lon = fake.longitude()
        
        transaction = {
            'transaction_id': i,
            'customer_id': customer_id,
            'merchant_id': merchant_id,
            'transaction_amount': round(amount, 2),
            'transaction_time': transaction_time,
            'transaction_location_lat': transaction_lat,
            'transaction_location_lon': transaction_lon
        }
        transactions.append(transaction)
        
        fraud_label = {
            'transaction_id': i,
            'is_fraud': int(is_fraud)
        }
        fraud_labels.append(fraud_label)
    
    return pd.DataFrame(transactions), pd.DataFrame(fraud_labels)

def main():
    """Generate all datasets and save to CSV files"""
    print("Generating customer data...")
    customers = generate_customers(500)
    
    print("Generating merchant data...")
    merchants = generate_merchants(100)
    
    print("Generating geolocation reference data...")
    geolocation_ref = generate_geolocation_reference(200)
    
    print("Generating transaction data and fraud labels...")
    transactions, fraud_labels = generate_transactions_and_fraud_labels(10000, 500, 100)
    
    # Save all datasets to CSV files
    print("Saving data to CSV files...")
    customers.to_csv('customers.csv', index=False)
    merchants.to_csv('merchants.csv', index=False)
    transactions.to_csv('transactions.csv', index=False)
    fraud_labels.to_csv('fraud_labels.csv', index=False)
    geolocation_ref.to_csv('geolocation_reference.csv', index=False)
    
    print("Data generation complete!")
    print(f"Generated {len(customers)} customers")
    print(f"Generated {len(merchants)} merchants")
    print(f"Generated {len(transactions)} transactions")
    print(f"Generated {len(fraud_labels)} fraud labels")
    print(f"Generated {len(geolocation_ref)} geolocation reference points")
    
    # Print fraud statistics
    fraud_rate = fraud_labels['is_fraud'].mean() * 100
    print(f"Fraud rate: {fraud_rate:.2f}%")
    
    # Display sample data
    print("\nSample customer data:")
    print(customers.head())
    
    print("\nSample merchant data:")
    print(merchants.head())
    
    print("\nSample transaction data:")
    print(transactions.head())
    
    print("\nSample fraud labels:")
    print(fraud_labels.head())
    
    print("\nSample geolocation reference:")
    print(geolocation_ref.head())

if __name__ == "__main__":
    main()