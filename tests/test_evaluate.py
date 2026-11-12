"""Tests for model evaluation and metrics output."""

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer


def test_model_evaluator(pipeline_data, tmp_path):
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline_data

    # Train a small model into a temp folder so the test doesn't need artifacts/
    trainer = ModelTrainer()
    trainer.config.model_file_path = tmp_path / "model.pkl"
    trainer.initiate_model_training(X_train, X_val, y_train, y_val, n_iter=2, cv=2)

    evaluator = ModelEvaluator()
    evaluator.config.trainer_config.model_file_path = trainer.config.model_file_path
    evaluator.config.metrics_path = tmp_path / "metrics.json"

    metrics = evaluator.initiate_model_evaluation(X_test, y_test)

    assert 0.75 < metrics["roc_auc"] <= 1.0
    assert 0.5 < metrics["pr_auc"] <= 1.0
    assert sum(map(sum, metrics["confusion_matrix"])) == len(y_test)
    assert evaluator.config.metrics_path.exists()
