# ============================================
# IMPORTING LIBRARIES
# ============================================

from fastapi import FastAPI
from app.schemas import ChurnRequest, ChurnResponse
from app.utils import model, prepare_input_data


# ============================================
# CREATING FASTAPI APP
# ============================================

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0",
    description="Predicts whether a customer is likely to churn using an XGBoost pipeline."
)


# ============================================
# HEALTH CHECK ENDPOINT
# ============================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================
# PREDICTION ENDPOINT
# ============================================

@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    input_data = prepare_input_data(request.model_dump())

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return ChurnResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4)
    )