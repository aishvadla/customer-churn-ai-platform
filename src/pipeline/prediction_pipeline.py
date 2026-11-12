"""Prediction pipeline for serving churn predictions with explanations."""

from pathlib import Path

import pandas as pd

from src.data.preprocessing import DataPreprocessingConfig
from src.models.evaluate import ModelEvaluatorConfig
from src.models.explain import ChurnExplainer
from src.models.train import ModelTrainerConfig
from src.utils.helper import load_object


class PredictionPipeline:
    """Load artifacts once; predict churn and explain each prediction."""

    def __init__(self, model_path=None, background_path=None):
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

    def predict(self, features: dict) -> dict:
        raw = pd.DataFrame([features])
        X = pd.DataFrame(
            self.preprocessor.transform(raw).astype("float32"),
            columns=self.feature_names,
        )
        proba = float(self.model.predict_proba(X.values)[0, 1])
        return {
            "churn_probability": round(proba, 4),
            "will_churn": proba >= self.threshold,
            "reasons": self.explainer.top_reasons(X, raw, self.original_cols)[0],
        }
