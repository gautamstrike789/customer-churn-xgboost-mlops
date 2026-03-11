# ============================================
# IMPORTING LIBRARIES
# ============================================

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app


# ============================================
# CREATING TEST CLIENT
# ============================================

client = TestClient(app)


# ============================================
# TEST HEALTH ENDPOINT
# ============================================

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ============================================
# TEST PREDICT ENDPOINT
# ============================================

def test_predict():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 1082.4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert "churn_prediction" in body
    assert "churn_probability" in body
    assert body["churn_prediction"] in [0, 1]
    assert 0.0 <= body["churn_probability"] <= 1.0