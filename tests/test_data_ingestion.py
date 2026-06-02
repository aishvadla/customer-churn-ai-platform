"""Tests for the data ingestion workflow."""

from src.data.ingestion import DataIngestion, DataIngestionConfig


def test_data_ingestion():
    val_size = 0.1
    test_size = 0.1
    random_state = 35
    config = DataIngestionConfig(
        val_size=val_size, test_size=test_size, random_state=random_state
    )
    ingestion_obj = DataIngestion(config)
    train_data_path, val_data_path, test_data_path = (
        ingestion_obj.initiate_data_ingestion()
    )

    assert train_data_path.exists(), f"Train data file does not exist"
    assert val_data_path.exists(), f"Validation data file does not exist"
    assert test_data_path.exists(), f"Test data file does not exist"
