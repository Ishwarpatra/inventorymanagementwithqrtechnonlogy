# Smart Inventory Management System

Revolutionizing retail business with a QR code-based smart inventory system for efficient stock management and real-time updates.

This project provides a comprehensive solution for modern retail, leveraging QR codes and AI to streamline inventory operations, prevent stockouts, and optimize stock replenishment.

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Challenges & Solutions](#challenges--solutions)
- [Project Status](#project-status)
- [Contact](#contact)

## About the Project
The Smart Inventory Management System, developed by Assassinators (Gyan Prakash (team leader), Koustubh Verma & Iswar Patra), is designed to simplify retail management through efficient stock tracking and real-time updates.

Key aspects include:
- **QR-BASED TRACKING**: Utilize QR codes for quick and accurate product identification, reducing manual errors.
- **REAL-TIME UPDATES**: Get instant stock updates and low-stock alerts to prevent overstocking or shortages.
- **SECURE ACCESS**: Role-based access control to protect sensitive inventory data.

## Features
Our system offers a seamless user experience, streamlining inventory tasks with the following core functionalities:

### QR Code Management
- Generate unique QR codes for products
- Scan QR codes for quick inventory access and tracking

### Stock Management
- Register initial stock, manufacturing, and expiry dates
- Monitor stock levels in real-time
- Update stock counts
- Track usage

### Product Management
- Add, update, and remove items from inventory
- Manage categories and brands

### AI-Powered Insights
- Predict sales based on item features
- Provide restocking suggestions based on AI insights
- Identify slow-moving and fast-moving products
- Suggest strategies to boost sales

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (Admin, Manager, Employee)
- Secure API endpoints

### Database Integration
- SQLite database for persistent storage
- Support for PostgreSQL (easily configurable)

## Technical Stack
- **Backend**: Python with FastAPI
- **Database**: SQLite (with PostgreSQL support)
- **Authentication**: JWT tokens with role-based access
- **Machine Learning**: XGBoost for sales prediction, K-Means for clustering
- **QR Codes**: Python qrcode library
- **Frontend**: Compatible with React/Angular/Vue.js (API-first design)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/inventorymanagementwithqrtechnology.git
```

2. Install dependencies:
```bash
pip install fastapi uvicorn pandas numpy xgboost scikit-learn qrcode[pil] python-jose[cryptography] passlib[bcrypt] sqlalchemy
```

3. Train the models:
```bash
cd backend/training
python train_model.py
python train_clustering.py
```

4. Run the application:
```bash
uvicorn app:app --reload
```

## Usage

1. Start the server: `uvicorn app:app --reload`
2. Access the API at `http://127.0.0.1:8000`
3. Use the `/docs` endpoint for API documentation and testing

### Default Credentials
- Admin: username: `admin`, password: `admin123`
- Manager: username: `manager`, password: `manager123`
- Employee: username: `employee`, password: `employee123`

## API Endpoints

### Authentication
- `POST /login` - Authenticate and get JWT token

### User Management
- `POST /users/create` - Create new user (Admin only)
- `GET /users/me` - Get current user info

### Inventory Management
- `GET /inventory` - Get all inventory items
- `GET /inventory/{item_id}` - Get specific item
- `POST /inventory` - Create new item (Manager+)
- `PUT /inventory/{item_id}` - Update item (Manager+)
- `DELETE /inventory/{item_id}` - Delete item (Admin only)

### QR Code Management
- `GET /generate_qr/{item_id}` - Generate QR code for item
- `POST /scan_qr` - Simulate QR code scanning

### AI-Powered Insights
- `POST /predict_sales` - Predict sales for an item
- `GET /recommend` - Get stock recommendations
- `GET /boost_sales` - Get sales boosting strategies

## Challenges & Solutions
We have identified and addressed several challenges:

### Challenge: Feature consistency in ML models
**Solution**: Ensured consistent feature engineering between training and inference phases.

### Challenge: Data preprocessing inconsistencies
**Solution**: Implemented standardized preprocessing and saved scalers for consistent transformation.

### Challenge: Authentication and authorization
**Solution**: Implemented JWT-based authentication with role-based access control.

### Challenge: Scalable data storage
**Solution**: Integrated SQL database with SQLAlchemy ORM for persistent storage.

## Project Status
This project is now complete with all core features implemented:
- ✅ QR code generation and scanning
- ✅ Machine learning models for sales prediction and recommendations
- ✅ Authentication and role-based access control
- ✅ Database integration
- ✅ RESTful API endpoints
- ✅ Proper error handling and logging

## Contact
For any inquiries or collaboration opportunities, please reach out to the team leaders:

Gyan Prakash: gyankngmpb1364@gmail.com,
Koustubh Verma: hp.koustubh@gmail.com,
Iswar Patra: patraishwar@yahoo.com,
