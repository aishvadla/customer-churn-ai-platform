"""Prediction pipeline for serving churn predictions with explanations."""

from pathlib import Path

import pandas as pd

from src.data.preprocessing import DataPreprocessingConfig
from src.models.evaluate import ModelEvaluatorConfig
from src.models.explain import ChurnExplainer
from src.models.train import ModelTrainerConfig
from src.utils.helper import load_object
from src.llm.retention_note import RetentionNoteGenerator


class PredictionPipeline:
    """Load artifacts once; predict churn and explain each prediction."""

    def __init__(self, model_path=None, background_path=None, note_generator=None):
        self.preprocessor = load_object(
            DataPreprocessingConfig().preprocessor_obj_file_path
        )
        self.model = load_object(model_path or ModelTrainerConfig().model_file_path)
        self.threshold = ModelEvaluatorConfig().threshold

        background = load_object(
            background_path or Path("artifacts") / "shap_background.pkl"
        )
        self.explainer = ChurnExplainer(self.model, background)

        self.feature_names = self.preprocessor.get_feature_names_out()
        # The raw columns the preprocessor actually uses, for grouping SHAP values
        self.original_cols = [
            col
            for name, _, cols in self.preprocessor.transformers_
            if name != "remainder"
            for col in cols
        ]

        self.note_generator = note_generator or RetentionNoteGenerator()

    def predict(self, features: dict, include_note: bool = True) -> dict:
        raw = pd.DataFrame([features])
        X = pd.DataFrame(
            self.preprocessor.transform(raw).astype("float32"),
            columns=self.feature_names,
        )
        proba = float(self.model.predict_proba(X.values)[0, 1])
        result = {
            "churn_probability": round(proba, 4),
            "will_churn": proba >= self.threshold,
            "reasons": self.explainer.top_reasons(X, raw, self.original_cols)[0],
        }
        # Only at-risk customers need a retention note (saves cost and latency)
        should_write = include_note and result["will_churn"]
        result["retention_note"] = (
            self.note_generator.generate(result) if should_write else None
        )
        return result


if __name__ == "__main__":
    import json

    customer = {
        "customerID": "DEMO-0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": "85.0",
    }
    print(json.dumps(PredictionPipeline().predict(customer), indent=2))
