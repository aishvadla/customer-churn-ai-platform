"""Tests for the data preprocessing workflow."""

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessing
import numpy as np


def test_data_preprocessing(pipeline_data):
    X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr = (
        pipeline_data
    )

    # Row count matches
    assert len(X_train_arr) == len(
        y_train_arr
    ), f"Training features and target row count don't match"
    assert len(X_val_arr) == len(
        y_val_arr
    ), f"Validation features and target row count don't match"
    assert len(X_test_arr) == len(
        y_test_arr
    ), f"Test features and target row count don't match"

    # No missing values after preprocessing
    assert not np.isnan(
        X_train_arr
    ).any(), f"Training features contain NaN after preprocessing"
    assert not np.isnan(
        X_val_arr
    ).any(), f"Validation features contain NaN after preprocessing"
    assert not np.isnan(
        X_test_arr
    ).any(), f"Test features contain NaN after preprocessing"

    # Correct dtypes for PyTorch
    assert X_train_arr.dtype == np.float32, f"Training array should be float32"
    assert X_val_arr.dtype == np.float32, f"Validation array should be float32"
    assert y_train_arr.dtype == np.float32, f"Training target array should be float32"

    # Shape sanity check
    assert (
        X_train_arr.shape[1] == X_val_arr.shape[1]
    ), f"Training and validation data feature count don't match"
    assert (
        X_train_arr.shape[1] == X_test_arr.shape[1]
    ), f"Training and test data feature count don't match"
