"""Training pipeline orchestration for the project.

This module connects data ingestion, preprocessing, model training,
and model evaluation into a single end-to-end pipeline.
"""

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessing
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.utils.logger import logger


class TrainPipeline:
    """Orchestrate the end-to-end model training and evaluation workflow."""

    def __init__(self):
        self.ingestion_obj = DataIngestion()
        self.preprocessor_obj = DataPreprocessing()
        self.trainer_obj = ModelTrainer()
        self.evaluator_obj = ModelEvaluator()

    def train(self):
        """Run the full training pipeline from ingestion through evaluation.

        Returns:
            dict: Final evaluation metrics produced by the model evaluator.
        """
        logger.info("Starting training pipeline")

        # Stage 1 — ingestion
        train_path, val_path, test_path = self.ingestion_obj.initiate_data_ingestion()

        # Stage 2 — preprocessing
        X_train, X_val, X_test, y_train, y_val, y_test = (
            self.preprocessor_obj.initiate_data_preprocessing(
                train_path, val_path, test_path
            )
        )

        # Stage 3 — training
        self.trainer_obj.initiate_model_training(X_train, X_val, y_train, y_val)

        # Stage 4 — evaluation
        metrics = self.evaluator_obj.initiate_model_evaluation(X_test, y_test)

        logger.info(f"Training pipeline complete. Metrics: {metrics}")
        return metrics
