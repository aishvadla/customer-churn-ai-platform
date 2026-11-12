"""Tests for the model training step."""

from src.models.train import ModelTrainer


def test_model_training(pipeline_data, tmp_path):
    X_train, X_val, _, y_train, y_val, _ = pipeline_data

    trainer = ModelTrainer()
    trainer.config.model_file_path = tmp_path / "model.pkl"

    best_name, best_auc = trainer.initiate_model_training(
        X_train, X_val, y_train, y_val, n_iter=2, cv=2
    )

    assert best_name in {"logistic_regression", "random_forest"}
    assert 0.75 < best_auc <= 1.0, f"Accuracy is not between 0.75 and 1"
    assert trainer.config.model_file_path.exists(), f"Model file does not exist"
