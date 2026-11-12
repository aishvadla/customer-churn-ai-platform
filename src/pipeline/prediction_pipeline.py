"""Prediction pipeline for serving churn predictions.

This module loads the preprocessing and model artifacts once and returns
a churn probability for a single customer record.
"""

import pandas as pd

from src.data.preprocessing import DataPreprocessingConfig
from src.models.evaluate import ModelEvaluatorConfig
from src.models.train import ModelTrainerConfig
from src.utils.helper import load_object


class PredictionPipeline:
    """Encapsulate inference logic for churn predictions."""

    def __init__(self, model_path=None):
        self.preprocessor = load_object(
            DataPreprocessingConfig().preprocessor_obj_file_path
        )
        self.model = load_object(model_path or ModelTrainerConfig().model_file_path)
        self.threshold = ModelEvaluatorConfig().threshold

    def predict(self, features: dict) -> dict:
        """Predict churn for a single customer feature dictionary.

        Parameters
        ----------
        features : dict
            Raw customer fields, keyed by the original Telco column names.

        Returns
        -------
        dict
            Churn probability and the yes/no decision at the configured threshold.
        """
        df = pd.DataFrame([features])
        X = self.preprocessor.transform(df).astype("float32")
        proba = float(self.model.predict_proba(X)[0, 1])
        return {
            "churn_probability": round(proba, 4),
            "will_churn": proba >= self.threshold,
        }
