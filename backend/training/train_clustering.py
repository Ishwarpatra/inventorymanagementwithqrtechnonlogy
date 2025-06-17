import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load Processed Data
df = pd.read_csv("processed_inventory.csv")

# Select Clustering Features
features = ['Stock Quantity', 'Sales Per Month']

# Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Train K-Means Model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Stock Cluster'] = kmeans.fit_predict(X_scaled)

# Save Model
joblib.dump(kmeans, "stock_cluster.pkl")
print("Clustering model saved successfully!")
