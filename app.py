from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, conint
from xgboost import XGBRegressor
from functools import lru_cache
import pandas as pd
import numpy as np
import joblib
import logging
import traceback
import os
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Configuration
CONFIG = {
    "model_path": os.path.join(os.path.dirname(__file__), "models", "inventory_model.pkl"),
    "cluster_model_path": os.path.join(os.path.dirname(__file__), "models", "stock_cluster.pkl"),
    "encoder_path": os.path.join(os.path.dirname(__file__), "models", "moencoders.pkl"),
    "data_path": os.path.join(os.path.dirname(__file__), "data", "processed_inventory.csv"),
    "cache_timeout": 3600  # 1 hour
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup and clean up on shutdown."""
    try:
        app.xgb_model = joblib.load(CONFIG["model_path"])
        app.kmeans_model = joblib.load(CONFIG["cluster_model_path"])
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")
        app.xgb_model, app.kmeans_model = None, None
    yield
    logger.info("Application shutdown completed.")

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def home():
    return {"message": "API is working!"}

# Cached data loading
@lru_cache(maxsize=1)
def load_processed_data():
    try:
        df = pd.read_parquet(CONFIG["data_path"])
        logger.info("Processed data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        return pd.DataFrame()

# Prediction request validation
from pydantic import Field

class PredictionRequest(BaseModel):
    Category: str
    Brand: str
    Stock_Quantity: int = Field(..., gt=0)
    Reorder_Level: int = Field(..., gt=0)
    Days_Since_Restock: int = Field(..., ge=0)

@app.get("/recommend")
def get_stock_recommendations():
    """Recommend which items to stock more or less based on sales trends."""
    try:
        df = load_processed_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Data not available")

        if not app.kmeans_model:
            raise HTTPException(status_code=500, detail="Clustering model not loaded")

        df['Stock Cluster'] = app.kmeans_model.predict(df[['Stock Quantity', 'Sales Per Month']])

        recommendations = []
        for _, row in df.iterrows():
            decision = "Stock More" if row['Sales Per Month'] > row['Stock Quantity'] else "Stock Less"
            reason = "High demand, low stock" if decision == "Stock More" else "Low sales, high stock"
            order_now = decision == "Stock More"

            recommendations.append({
                "Item": row['Item Name'],
                "Stock Quantity": row['Stock Quantity'],
                "Sales Per Month": row['Sales Per Month'],
                "Decision": decision,
                "Reason": reason,
                "Order Now": order_now
            })

        return {"recommendations": recommendations}

    except Exception as e:
        logger.error(f"Recommendation error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Recommendation failed")

@app.get("/boost_sales")
def boost_sales():
    """Suggest strategies to boost sales using ML insights."""
    try:
        df = load_processed_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Data not available")

        if not app.xgb_model:
            raise HTTPException(status_code=500, detail="Prediction model not loaded")

        q75 = df['Sales Per Month'].quantile(0.75)
        q25 = df['Sales Per Month'].quantile(0.25)

        fast_moving = df[df['Sales Per Month'] > q75]
        slow_moving = df[df['Sales Per Month'] < q25]

        boost_suggestions = []
        for _, row in slow_moving.iterrows():
            input_data = np.array([[row["Stock Quantity"], row["Sales Per Month"]]])
            predicted_sales = app.xgb_model.predict(input_data)[0]
            strategy = (
                "Discount promotions" if predicted_sales > row["Sales Per Month"]
                else "Improve visibility in store"
            )

            boost_suggestions.append({
                "Item": row["Item Name"],
                "Current Sales": row["Sales Per Month"],
                "Predicted Sales After Strategy": float(predicted_sales),
                "Recommended Strategy": strategy
            })

        return {
            "fast_moving": fast_moving[['Item Name', 'Sales Per Month']].to_dict(orient='records'),
            "slow_moving": slow_moving[['Item Name', 'Sales Per Month']].to_dict(orient='records'),
            "boost_suggestions": boost_suggestions
        }

    except Exception as e:
        logger.error(f"Boost sales error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Sales boost analysis failed")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)