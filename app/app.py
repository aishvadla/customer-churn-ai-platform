"""FastAPI application for churn prediction.

Exposes the trained churn model as a REST API: validates customer data,
returns churn probability, top reasons and a retention note if at risk.
"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.pipeline.prediction_pipeline import PredictionPipeline

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0)


class Reason(BaseModel):
    feature: str
    value: str | int | float
    impact: float
    effect: str


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    reasons: list[Reason]
    retention_note: str | None


app = FastAPI(title="Churn AI")
pipeline = PredictionPipeline()  # load model, preprocessor and SHAP data once


@app.get("/")
def root():
    """Point visitors to the interactive API docs."""
    return {"message": "Churn AI prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    """Return application health status."""
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    """Predict churn for one customer, with top reasons and a retention note if at risk."""
    return pipeline.predict(customer.model_dump())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
