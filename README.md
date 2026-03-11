# Customer Churn Prediction MLOps Pipeline

## Live API Demo

API Documentation:

---

## Project Overview

This project demonstrates an end-to-end **Machine Learning Operations (MLOps) pipeline** for predicting customer churn using an XGBoost model. The system includes model training, artifact generation, an inference API built with FastAPI, Docker containerization, and automated CI/CD using GitHub Actions.

The goal is to simulate a production-ready machine learning workflow rather than a simple notebook experiment.

---

## Key Features

• Customer churn prediction using **XGBoost**
• Automated **data preprocessing and feature engineering**
• **Model evaluation metrics** including Accuracy, Precision, Recall, F1, and ROC-AUC
• **FastAPI inference service** for real-time predictions
• **Docker containerization** for reproducible deployment
• **Pytest API tests** for validation
• **GitHub Actions CI/CD pipeline** that runs tests and Docker builds automatically

---

## Project Architecture

Dataset
↓
Training Pipeline (`src/train.py`)
↓
Saved Model Artifacts
↓
FastAPI Prediction Service (`app/`)
↓
Docker Container
↓
CI/CD with GitHub Actions

---

## Project Structure

```
customer-churn-xgboost-mlops

app/
main.py
schemas.py
utils.py

src/
train.py

tests/
test_api.py

artifacts/
model.joblib
feature_columns.joblib

data/
Telco-Customer-Churn.csv

.github/workflows/
ci.yml

Dockerfile
requirements.txt
README.md
```

---

## Model Performance

After training the XGBoost model:

Accuracy: ~0.75
Precision: ~0.52
Recall: ~0.81
F1 Score: ~0.63
ROC-AUC: ~0.84

The model performs well in detecting churn cases with strong recall.

---

## Running the Training Pipeline

Train the model and generate artifacts:

```
python src/train.py
```

Artifacts generated:

```
artifacts/model.joblib
artifacts/feature_columns.joblib
```

These files are used by the FastAPI service for inference.

---

## Running the API Locally

Start the FastAPI server:

```
uvicorn app.main:app --reload
```

Open API documentation:

```
http://127.0.0.1:8000/docs
```

You can test predictions directly through the interactive Swagger UI.

---

## Example Prediction Request

```
{
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
```

Example response:

```
{
"prediction": 1,
"probability": 0.73
}
```

---

## Running with Docker

Build Docker image:

```
docker build -t churn-api .
```

Run container:

```
docker run -p 8000:8000 churn-api
```

Then open:

```
http://localhost:8000/docs
```

---

## Automated CI/CD Pipeline

GitHub Actions automatically runs on every push.

Pipeline steps:

1. Install dependencies
2. Run automated API tests with pytest
3. Build Docker image

This ensures the application is always testable and deployable.

---

## Tech Stack

Python
XGBoost
Scikit-learn
FastAPI
Docker
GitHub Actions
Pytest

---

## Dataset

Telco Customer Churn Dataset
Contains customer demographics, account information, and service usage details used to predict churn behavior.

---

## Author

Aman Sharma
Data Scientist
