"""Tests for the end-to-end prediction pipeline."""

import pandas as pd

from src.data.preprocessing import DataPreprocessingConfig
from src.models.train import ModelTrainer
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.helper import load_object, save_object

BASE_CUSTOMER = {
    "customerID": "TEST-0001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
}

HIGH_RISK = {
    **BASE_CUSTOMER,
    "tenure": 1,
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": "85.0",
}

LOW_RISK = {
    **BASE_CUSTOMER,
    "tenure": 70,
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "Contract": "Two year",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": "3850.0",
}


def test_prediction_pipeline(pipeline_data, tmp_path):
    X_train, X_val, _, y_train, y_val, _ = pipeline_data

    # Train a small model into a temp folder (no dependency on artifacts/model.pkl)
    trainer = ModelTrainer()
    trainer.config.model_file_path = tmp_path / "model.pkl"
    trainer.initiate_model_training(X_train, X_val, y_train, y_val, n_iter=2, cv=2)

    # Build the SHAP background sample in the temp folder too
    preprocessor = load_object(DataPreprocessingConfig().preprocessor_obj_file_path)
    background = pd.DataFrame(
        X_train, columns=preprocessor.get_feature_names_out()
    ).sample(50, random_state=42)
    background_path = tmp_path / "background.pkl"
    save_object(obj=background, file_path=background_path)

    pipeline = PredictionPipeline(
        model_path=trainer.config.model_file_path,
        background_path=background_path,
    )
    high = pipeline.predict(HIGH_RISK)
    low = pipeline.predict(LOW_RISK)

    # Output format
    for result in (high, low):
        assert 0.0 <= result["churn_probability"] <= 1.0
        assert isinstance(result["will_churn"], bool)
        assert len(result["reasons"]) == 3
        for reason in result["reasons"]:
            assert set(reason) == {"feature", "value", "impact", "effect"}

    # Behaviour: risky customer scores higher, and the explanations point the right way
    assert high["churn_probability"] > low["churn_probability"]
    assert high["reasons"][0]["effect"] == "increases churn risk"
    assert low["reasons"][0]["effect"] == "decreases churn risk"
