import requests

# Test the API
base_url = "http://127.0.0.1:8000"

# Test home endpoint
try:
    response = requests.get(f"{base_url}/")
    print("Home endpoint:", response.json())
except Exception as e:
    print(f"Error connecting to home endpoint: {e}")

# Test docs endpoint
try:
    response = requests.get(f"{base_url}/docs")
    print("Docs endpoint status:", response.status_code)
except Exception as e:
    print(f"Error connecting to docs endpoint: {e}")