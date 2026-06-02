"""Data preprocessing module for the auto fuel efficiency pipeline.

This module defines preprocessing logic for numeric and categorical
features and persists the preprocessor artifacts for later prediction.
"""

from pathlib import Path
from src.utils.logger import logger
from src.utils.exception import CustomException
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.helper import save_object
import sys
import numpy as np


@dataclass
class DataPreprocessingConfig:
    """Configuration for preprocessing artifact file paths."""

    preprocessor_obj_file_path = Path("artifacts") / "preprocessor.pkl"
    scaler_y_obj_file_path = Path("artifacts") / "scaler_y.pkl"


class DataPreprocessing:
    """Create preprocessing pipelines and transform raw datasets."""

    def __init__(self):
        """Initialize preprocessing configuration."""
        self.data_preprocessing_config = DataPreprocessingConfig()

    def initiate_data_preprocessing(
        self, train_data_path, val_data_path, test_data_path
    ):
        """Load datasets, create feature pipelines, and transform all splits.

        Parameters
        ----------
        train_data_path : str | Path
            Path to the training CSV file.
        val_data_path : str | Path
            Path to the validation CSV file.
        test_data_path : str | Path
            Path to the test CSV file.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Transformed training, validation, and test features and targets.
        """
        try:
            logger.info("Initiated Data Preprocessing")
            df_train = pd.read_csv(train_data_path)
            df_val = pd.read_csv(val_data_path)
            df_test = pd.read_csv(test_data_path)

            # Identify numeric and categorical features and target
            # column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
            target_column = "MPG"
            # Using Model Year as a numeric column instead of ordinal to avoid the equal spacing between years
            numeric_mean_columns = [
                "Displacement",
                "Horsepower",
                "Weight",
                "Acceleration",
            ]
            numeric_mode_columns = ["Cylinders", "Model Year"]
            categorical_nominal_columns = ["Origin"]

            df_y_train = df_train[target_column]
            df_X_train = df_train.drop(columns=[target_column])
            df_y_val = df_val[target_column]
            df_X_val = df_val.drop(columns=[target_column])
            df_y_test = df_test[target_column]
            df_X_test = df_test.drop(columns=[target_column])

            logger.info("Creating preprocessing pipeline object")
            # Preprocessing pipeline
            numeric_mean_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )
            numeric_mode_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler()),
                ]
            )
            nominal_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore", drop="first", sparse_output=False
                        ),
                    ),
                ]
            )

            preprocessing_pipeline = ColumnTransformer(
                [
                    ("numeric_mean", numeric_mean_pipeline, numeric_mean_columns),
                    ("numeric_mode", numeric_mode_pipeline, numeric_mode_columns),
                    ("nominal", nominal_pipeline, categorical_nominal_columns),
                ]
            )

            # fit and transform training data
            X_train_arr = preprocessing_pipeline.fit_transform(df_X_train)

            # transform validation and test data
            X_val_arr = preprocessing_pipeline.transform(df_X_val)
            X_test_arr = preprocessing_pipeline.transform(df_X_test)

            # Scale target
            scaler_y = StandardScaler()
            y_train_arr = (
                scaler_y.fit_transform(df_y_train.values.reshape(-1, 1))
                .flatten()
                .astype("float32")
            )
            y_val_arr = (
                scaler_y.transform(df_y_val.values.reshape(-1, 1))
                .flatten()
                .astype("float32")
            )

            y_test_arr = (
                scaler_y.transform(df_y_test.values.reshape(-1, 1))
                .flatten()
                .astype("float32")
            )

            # save preprocessor objects
            save_object(
                obj=preprocessing_pipeline,
                file_path=self.data_preprocessing_config.preprocessor_obj_file_path,
            )
            save_object(
                obj=scaler_y,
                file_path=self.data_preprocessing_config.scaler_y_obj_file_path,
            )
            logger.info("Saved preprocessing pipeline objects")

            logger.info("Data Transformation Completed")
            return (
                X_train_arr.astype("float32"),
                X_val_arr.astype("float32"),
                X_test_arr.astype("float32"),
                y_train_arr,
                y_val_arr,
                y_test_arr,
            )
        except Exception as e:
            raise CustomException(e, sys)
