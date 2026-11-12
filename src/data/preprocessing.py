"""Data preprocessing pipeline for the customer churn engine."""

import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.helper import save_object, load_config


def collapse_no_service(X):
    """Map 'No internet service' / 'No phone service' to 'No'."""
    return pd.DataFrame(X).replace(
        {"No internet service": "No", "No phone service": "No"}
    )


def to_numeric(X):
    """Convert columns to numbers; blanks (e.g. TotalCharges for new customers) become NaN."""
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")


@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_file_path: Path = Path("artifacts") / "preprocessor.pkl"


class DataPreprocessing:
    def __init__(self):
        self.config = DataPreprocessingConfig()

    def initiate_data_preprocessing(self, train_path, val_path, test_path):
        logger.info("Initiated data preprocessing")
        try:
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            df_test = pd.read_csv(test_path)

            # Drop ID column
            for df in [df_train, df_val, df_test]:
                df.drop(columns=["customerID"], inplace=True)
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

            # Target encoding
            target = "Churn"
            for df in [df_train, df_val, df_test]:
                df[target] = df[target].map({"Yes": 1, "No": 0})

            # Split X and y
            y_train = df_train[target].values.astype("float32")
            y_val = df_val[target].values.astype("float32")
            y_test = df_test[target].values.astype("float32")

            X_train = df_train.drop(columns=[target])
            X_val = df_val.drop(columns=[target])
            X_test = df_test.drop(columns=[target])

            # Feature groups
            numeric_cols = ["tenure", "MonthlyCharges"]
            binary_cols = [
                "Partner",
                "Dependents",
                "PhoneService",
                "PaperlessBilling",
                "MultipleLines",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
            nominal_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]

            # Pipelines
            numeric_pipeline = Pipeline(
                [
                    (
                        "to_numeric",
                        FunctionTransformer(to_numeric, feature_names_out="one-to-one"),
                    ),
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            binary_pipeline = Pipeline(
                [
                    (
                        "collapse",
                        FunctionTransformer(
                            collapse_no_service, feature_names_out="one-to-one"
                        ),
                    ),
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            categories=[["No", "Yes"]] * len(binary_cols),
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )

            nominal_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            drop="first", handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numeric", numeric_pipeline, numeric_cols),
                    ("binary", binary_pipeline, binary_cols),
                    ("nominal", nominal_pipeline, nominal_cols),
                    ("passthrough", "passthrough", ["SeniorCitizen"]),
                ],
                remainder="drop",
            )

            # Fit on train only
            preprocessor.fit(X_train)
            X_train_arr = preprocessor.transform(X_train).astype("float32")
            X_val_arr = preprocessor.transform(X_val).astype("float32")
            X_test_arr = preprocessor.transform(X_test).astype("float32")

            # Save preprocessor
            Path("artifacts").mkdir(parents=True, exist_ok=True)
            save_object(
                obj=preprocessor, file_path=self.config.preprocessor_obj_file_path
            )
            logger.info("Saved preprocessor to artifacts")

            logger.info("Data preprocessing complete")
            return X_train_arr, X_val_arr, X_test_arr, y_train, y_val, y_test

        except Exception as e:
            raise CustomException(e, sys)
