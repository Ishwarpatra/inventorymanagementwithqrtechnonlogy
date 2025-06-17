import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "indian_retail_inventory.csv")
model_dir = os.path.join(BASE_DIR, "..", "models")
processed_path = os.path.join(BASE_DIR, "..", "data", "processed_inventory.csv")

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(csv_path)
df.ffill(inplace=True)


# Convert date columns
df['Expiry Date'] = pd.to_datetime(df['Expiry Date'], errors='coerce')
df['Last Restocked Date'] = pd.to_datetime(df['Last Restocked Date'], errors='coerce')
df['Days Since Restock'] = (pd.to_datetime('today') - df['Last Restocked Date']).dt.days

# Encode categorical features
label_enc_category = LabelEncoder()
label_enc_brand = LabelEncoder()
df['Category'] = label_enc_category.fit_transform(df['Category'].astype(str))
df['Brand'] = label_enc_brand.fit_transform(df['Brand'].astype(str))

# Define training features and target
X = df[['Category', 'Brand', 'Stock Quantity', 'Reorder Level', 'Days Since Restock']]
y = df['Sales Per Month']

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
xgb_model.fit(X, y)

# Save model
joblib.dump(xgb_model, os.path.join(model_dir, 'inventory_model.pkl'))

# Save encoders
joblib.dump({
    'category': label_enc_category,
    'brand': label_enc_brand
}, os.path.join(model_dir, 'moencoders.pkl'))

# Train clustering model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Stock Cluster'] = kmeans.fit_predict(df[['Stock Quantity', 'Sales Per Month']])

# Save clustering model
joblib.dump(kmeans, os.path.join(model_dir, 'stock_cluster.pkl'))

# Save processed inventory
df.to_csv(processed_path, index=False)

print("✅ Model training complete. Files saved in /models and /data")
