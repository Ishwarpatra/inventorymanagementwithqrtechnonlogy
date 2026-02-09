from fastapi import FastAPI, HTTPException, Query, Depends
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
import qrcode
from io import BytesIO
import base64
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Database imports
from sqlalchemy.orm import Session
from models import engine, SessionLocal, InventoryItem, User as DBUser

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = "your-secret-key-change-this-in-production"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token bearer scheme
security = HTTPBearer()

# User roles
USER_ROLES = {
    "admin": ["admin", "manager", "employee"],
    "manager": ["manager", "employee"],
    "employee": ["employee"]
}

# Configuration
CONFIG = {
    "model_path": os.path.join(os.path.dirname(__file__), "backend", "models", "inventory_model.pkl"),
    "cluster_model_path": os.path.join(os.path.dirname(__file__), "backend", "models", "stock_cluster.pkl"),
    "encoder_path": os.path.join(os.path.dirname(__file__), "backend", "models", "moencoders.pkl"),
    "data_path": os.path.join(os.path.dirname(__file__), "backend", "data", "processed_inventory.csv"),  # Fixed: Changed from parquet to csv
    "scaler_path": os.path.join(os.path.dirname(__file__), "backend", "models", "scaler.pkl"),  # Added scaler path
    "cache_timeout": 3600  # 1 hour
}


# User model
class User(BaseModel):
    username: str
    email: Optional[str] = None
    role: str = "employee"  # Default role


# Token model
class Token(BaseModel):
    access_token: str
    token_type: str


# Token data model
class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize mock user database with hashed passwords
def initialize_mock_users():
    # Create users with properly truncated passwords
    users = {}
    users["admin"] = {
        "username": "admin",
        "email": "admin@example.com",
        "role": "admin",
        "hashed_password": pwd_context.hash("admin123"),  # Default admin password
    }
    users["manager"] = {
        "username": "manager",
        "email": "manager@example.com",
        "role": "manager",
        "hashed_password": pwd_context.hash("mgr123"),  # Default manager password
    }
    users["employee"] = {
        "username": "employee",
        "email": "employee@example.com",
        "role": "employee",
        "hashed_password": pwd_context.hash("emp123"),  # Default employee password
    }
    return users

# Initialize the users database later to avoid bcrypt initialization issues
fake_users_db = {}

def verify_password(plain_password, hashed_password):
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Hash a plain password."""
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str):
    """Authenticate a user by username and password."""
    user = fake_users_db.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get the current authenticated user from the token."""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception
    
    user = fake_users_db.get(token_data.username)
    if user is None:
        raise credentials_exception
    return user


def require_role(required_role: str):
    """Dependency to check if user has required role."""
    async def role_checker(current_user: dict = Depends(get_current_user)):
        user_role = current_user.get("role")
        if user_role not in USER_ROLES.get(required_role, []):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required role: {required_role}, User role: {user_role}"
            )
        return current_user
    return role_checker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup and clean up on shutdown."""
    global fake_users_db  # Initialize the global variable here
    try:
        app.xgb_model = joblib.load(CONFIG["model_path"])
        app.kmeans_model = joblib.load(CONFIG["cluster_model_path"])
        app.encoders = joblib.load(CONFIG["encoder_path"])
        # Try to load scaler, if not available, we'll handle it later
        try:
            app.scaler = joblib.load(CONFIG["scaler_path"])
        except:
            app.scaler = None
            logger.warning("Scaler not found. Clustering may not work properly.")
        
        # Initialize the fake users database here to avoid bcrypt initialization issues
        fake_users_db.update(initialize_mock_users())
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")
        app.xgb_model, app.kmeans_model, app.encoders = None, None, None
    yield
    logger.info("Application shutdown completed.")


# Define the lifespan function first
async def startup_event():
    """Initialize models and users at startup."""
    global fake_users_db  # Initialize the global variable here
    try:
        app.xgb_model = joblib.load(CONFIG["model_path"])
        app.kmeans_model = joblib.load(CONFIG["cluster_model_path"])
        app.encoders = joblib.load(CONFIG["encoder_path"])
        # Try to load scaler, if not available, we'll handle it later
        try:
            app.scaler = joblib.load(CONFIG["scaler_path"])
        except:
            app.scaler = None
            logger.warning("Scaler not found. Clustering may not work properly.")
        
        # Initialize the fake users database here to avoid bcrypt initialization issues
        fake_users_db.update(initialize_mock_users())
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")
        app.xgb_model, app.kmeans_model, app.encoders = None, None, None

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Login endpoint
@app.post("/login", response_model=Token)
def login(username: str, password: str):
    """Login endpoint to get access token."""
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Add event handlers
app.on_event("startup")(startup_event)

@app.get("/")
def home():
    return {"message": "API is working!"}

# Cached data loading
@lru_cache(maxsize=1)
def load_processed_data():
    """
    Load processed data from database instead of CSV.
    This function creates a DataFrame from the database records for ML processing.
    """
    try:
        # Create a temporary session to fetch data
        db = SessionLocal()
        try:
            # Query all active inventory items from the database
            db_items = db.query(InventoryItem).filter(InventoryItem.is_active == True).all()
            
            # Convert to DataFrame format expected by ML models
            if db_items:
                data_list = []
                for item in db_items:
                    data_list.append({
                        'Item Name': item.item_name,
                        'Category': item.category,
                        'Brand': item.brand,
                        'Stock Quantity': item.stock_quantity,
                        'Reorder Level': item.reorder_level,
                        'Days Since Restock': item.days_since_restock,
                        'Sales Per Month': item.sales_per_month,
                        'Expiry Date': item.expiry_date,
                        'Last Restocked Date': item.last_restocked_date,
                        'QR Code': item.qr_code
                    })
                
                df = pd.DataFrame(data_list)
                logger.info(f"Processed data loaded successfully from database: {len(df)} items")
                return df
            else:
                logger.warning("No items found in database")
                return pd.DataFrame()
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error loading processed data from database: {str(e)}")
        # Fallback to CSV if database fails
        try:
            df = pd.read_csv(CONFIG["data_path"])
            logger.info("Loaded data from CSV as fallback")
            return df
        except:
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
def get_stock_recommendations(current_user: dict = Depends(get_current_user)):
    """Recommend which items to stock more or less based on sales trends."""
    try:
        df = load_processed_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Data not available")

        if not app.kmeans_model:
            raise HTTPException(status_code=500, detail="Clustering model not loaded")

        # Apply scaling if scaler is available (for consistency with training)
        if app.scaler:
            cluster_features = app.scaler.transform(df[['Stock Quantity', 'Sales Per Month']])
            df['Stock Cluster'] = app.kmeans_model.predict(cluster_features)
        else:
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

@app.post("/predict_sales")
def predict_sales(request: PredictionRequest, current_user: dict = Depends(get_current_user)):
    """Predict sales based on item features using the trained XGBoost model."""
    try:
        if not app.xgb_model or not app.encoders:
            raise HTTPException(status_code=500, detail="Models not loaded properly")

        # Encode categorical features using the saved encoders
        try:
            category_encoded = app.encoders['category'].transform([request.Category])[0]
        except ValueError:
            # Handle unseen categories by using the most frequent category encoding
            category_encoded = app.encoders['category'].classes_[0]  # Use first class as fallback
        
        try:
            brand_encoded = app.encoders['brand'].transform([request.Brand])[0]
        except ValueError:
            # Handle unseen brands by using the most frequent brand encoding
            brand_encoded = app.encoders['brand'].classes_[0]  # Use first class as fallback

        # Prepare input data with all required features (same as training)
        input_data = np.array([[category_encoded, brand_encoded, 
                                request.Stock_Quantity, 
                                request.Reorder_Level, 
                                request.Days_Since_Restock]])

        # Make prediction
        predicted_sales = app.xgb_model.predict(input_data)[0]

        return {
            "predicted_sales": float(predicted_sales),
            "input_features": {
                "category_encoded": int(category_encoded),
                "brand_encoded": int(brand_encoded),
                "stock_quantity": request.Stock_Quantity,
                "reorder_level": request.Reorder_Level,
                "days_since_restock": request.Days_Since_Restock
            }
        }

    except Exception as e:
        logger.error(f"Sales prediction error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Sales prediction failed: {str(e)}")

@app.get("/boost_sales")
def boost_sales(current_user: dict = Depends(get_current_user)):
    """Suggest strategies to boost sales using ML insights."""
    try:
        df = load_processed_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Data not available")

        if not app.xgb_model or not app.encoders:
            raise HTTPException(status_code=500, detail="Prediction model or encoders not loaded")

        q75 = df['Sales Per Month'].quantile(0.75)
        q25 = df['Sales Per Month'].quantile(0.25)

        fast_moving = df[df['Sales Per Month'] > q75]
        slow_moving = df[df['Sales Per Month'] < q25]

        boost_suggestions = []
        for _, row in slow_moving.iterrows():
            # Prepare input data with all required features (same as training)
            # Need to decode the encoded values back to original to get the encodings
            try:
                category_encoded = int(row['Category'])  # Assuming this is already encoded
                brand_encoded = int(row['Brand'])  # Assuming this is already encoded
            except:
                # If the values are not encoded, we need to encode them
                try:
                    category_encoded = app.encoders['category'].transform([str(row['Category'])])[0]
                except ValueError:
                    category_encoded = app.encoders['category'].classes_[0]
                
                try:
                    brand_encoded = app.encoders['brand'].transform([str(row['Brand'])])[0]
                except ValueError:
                    brand_encoded = app.encoders['brand'].classes_[0]

            input_data = np.array([[category_encoded, brand_encoded, 
                                    row["Stock Quantity"], 
                                    row["Reorder Level"], 
                                    row["Days Since Restock"]]])
            
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

# QR Code Generation Endpoint
@app.get("/generate_qr/{item_id}")
def generate_qr(item_id: str, current_user: dict = Depends(get_current_user)):
    """Generate a QR code for a specific item ID."""
    try:
        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(item_id)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        
        # Encode to base64 for transmission
        img_base64 = base64.b64encode(img_bytes).decode()
        
        return {
            "item_id": item_id,
            "qr_code": f"data:image/png;base64,{img_base64}"
        }
    except Exception as e:
        logger.error(f"QR generation error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="QR code generation failed")

# QR Code Scanner Simulation (in a real app, this would process scanned data)
@app.post("/scan_qr")
def scan_qr(item_id: str = Query(..., description="Item ID from QR code"), current_user: dict = Depends(get_current_user)):
    """Simulate scanning a QR code and returning item information."""
    try:
        df = load_processed_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Data not available")

        # Find item by ID (assuming Item ID column exists, otherwise use Item Name)
        item_info = df[df['Item Name'] == item_id]  # Adjust column name as needed
        
        if item_info.empty:
            # If not found by name, try to match by other fields
            item_info = df[df.astype(str).apply(lambda x: x.str.contains(item_id, na=False)).any(axis=1)]
        
        if item_info.empty:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Return the first matching item
        item = item_info.iloc[0].to_dict()
        
        return {
            "scanned_item_id": item_id,
            "item_details": item
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"QR scan simulation error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="QR scan processing failed")

# User management endpoints
@app.post("/users/create")
def create_user(user: User, current_user: dict = Depends(require_role("admin"))):
    """Create a new user (admin only)."""
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    fake_users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "hashed_password": get_password_hash("newuser123")  # Default password for new users
    }
    
    return {"message": f"User {user.username} created successfully"}


@app.get("/users/me")
def read_users_me(current_user: dict = Depends(get_current_user)):
    """Get current user info."""
    return current_user


# Inventory management endpoints
@app.get("/inventory")
def get_inventory(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """Get all inventory items."""
    inventory_items = db.query(InventoryItem).filter(InventoryItem.is_active == True).all()
    return {"inventory": inventory_items}


@app.get("/inventory/{item_id}")
def get_inventory_item(item_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """Get a specific inventory item by ID."""
    item = db.query(InventoryItem).filter(InventoryItem.id == item_id, InventoryItem.is_active == True).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.post("/inventory")
def create_inventory_item(item: dict, db: Session = Depends(get_db), current_user: dict = Depends(require_role("manager"))):
    """Create a new inventory item."""
    # Create a new inventory item from the dictionary
    db_item = InventoryItem(
        item_name=item.get('item_name'),
        category=item.get('category'),
        brand=item.get('brand'),
        stock_quantity=item.get('stock_quantity', 0),
        reorder_level=item.get('reorder_level', 10),
        expiry_date=pd.to_datetime(item.get('expiry_date')) if item.get('expiry_date') else None,
        last_restocked_date=pd.to_datetime(item.get('last_restocked_date')) if item.get('last_restocked_date') else None,
        days_since_restock=item.get('days_since_restock', 0),
        sales_per_month=item.get('sales_per_month', 0.0),
        qr_code=item.get('qr_code')
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


@app.put("/inventory/{item_id}")
def update_inventory_item(item_id: int, item: dict, db: Session = Depends(get_db), current_user: dict = Depends(require_role("manager"))):
    """Update an existing inventory item."""
    db_item = db.query(InventoryItem).filter(InventoryItem.id == item_id, InventoryItem.is_active == True).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Update fields
    if 'item_name' in item:
        db_item.item_name = item['item_name']
    if 'category' in item:
        db_item.category = item['category']
    if 'brand' in item:
        db_item.brand = item['brand']
    if 'stock_quantity' in item:
        db_item.stock_quantity = item['stock_quantity']
    if 'reorder_level' in item:
        db_item.reorder_level = item['reorder_level']
    if 'expiry_date' in item:
        db_item.expiry_date = pd.to_datetime(item['expiry_date']) if item['expiry_date'] else None
    if 'last_restocked_date' in item:
        db_item.last_restocked_date = pd.to_datetime(item['last_restocked_date']) if item['last_restocked_date'] else None
    if 'days_since_restock' in item:
        db_item.days_since_restock = item['days_since_restock']
    if 'sales_per_month' in item:
        db_item.sales_per_month = item['sales_per_month']
    if 'qr_code' in item:
        db_item.qr_code = item['qr_code']
    
    db.commit()
    return db_item


@app.delete("/inventory/{item_id}")
def delete_inventory_item(item_id: int, db: Session = Depends(get_db), current_user: dict = Depends(require_role("admin"))):
    """Soft delete an inventory item."""
    db_item = db.query(InventoryItem).filter(InventoryItem.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Soft delete by setting is_active to False
    db_item.is_active = False
    db.commit()
    return {"message": "Item deleted successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)