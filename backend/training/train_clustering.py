import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "indian_retail_inventory.csv")
model_dir = os.path.join(BASE_DIR, "..", "models")
processed_path = os.path.join(BASE_DIR, "..", "data", "processed_inventory.csv")

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Load Processed Data
df = pd.read_csv(processed_path)

# Select Clustering Features
features = ['Stock Quantity', 'Sales Per Month']

# Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Train K-Means Model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Stock Cluster'] = kmeans.fit_predict(X_scaled)

# Save Models
joblib.dump(kmeans, os.path.join(model_dir, "stock_cluster.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))  # Save the scaler too
print("Clustering model and scaler saved successfully!")
