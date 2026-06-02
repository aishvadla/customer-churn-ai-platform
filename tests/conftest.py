"""PyTest fixture definitions for end-to-end pipeline testing."""

import pytest
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessing


@pytest.fixture(scope="session")
def pipeline_data():
    """Runs once per test session, shared across all tests."""
    ingestion_obj = DataIngestion()
    preprocessing_obj = DataPreprocessing()

    train_path, val_path, test_path = ingestion_obj.initiate_data_ingestion()
    X_train, X_val, X_test, y_train, y_val, y_test = (
        preprocessing_obj.initiate_data_preprocessing(train_path, val_path, test_path)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
