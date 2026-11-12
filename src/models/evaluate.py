"""Model evaluation utilities for the customer churn pipeline.

This module loads the saved model, evaluates performance on the held-out
test dataset, and writes metrics to disk.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.models.train import ModelTrainerConfig
from src.utils.exception import CustomException
from src.utils.helper import load_object
from src.utils.logger import logger


@dataclass
class ModelEvaluatorConfig:
    """Configuration for evaluation artifact locations."""

    trainer_config: ModelTrainerConfig = field(default_factory=ModelTrainerConfig)
    metrics_path: Path = Path("artifacts") / "metrics.json"
    threshold: float = 0.5


class ModelEvaluator:
    """Evaluate a trained model on the held-out test dataset."""

    def __init__(self):
        """Initialize the evaluator with configuration."""
        self.config = ModelEvaluatorConfig()

    def initiate_model_evaluation(self, X_test, y_test):
        """Evaluate the model and persist metrics.

        Parameters
        ----------
        X_test : np.ndarray
            Preprocessed test features.
        y_test : np.ndarray
            Test labels (1 = churn, 0 = no churn).

        Returns
        -------
        dict
            ROC-AUC, PR-AUC, confusion matrix and per-class report.
        """
        logger.info("Initiated model evaluation on test set")
        try:
            model = load_object(self.config.trainer_config.model_file_path)

            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba >= self.config.threshold).astype(int)

            metrics = {
                "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
                "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
                "threshold": self.config.threshold,
                "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
                "report": classification_report(
                    y_test, preds, target_names=["no_churn", "churn"], output_dict=True
                ),
            }

            self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(
                f"Test ROC-AUC: {metrics['roc_auc']}, PR-AUC: {metrics['pr_auc']}, "
                f"churn recall: {metrics['report']['churn']['recall']:.4f}"
            )
            logger.info("Saved metrics.json file to artifacts")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)
