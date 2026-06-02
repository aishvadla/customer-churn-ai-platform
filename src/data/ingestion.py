"""Data ingestion utilities for the customer churn engine pipeline.

This module downloads the Telco-Customer-Churn dataset, writes raw CSV files,
and performs train/validation/test splits.
"""

import sys
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from src.utils.logger import logger
from src.utils.exception import CustomException
from sklearn.model_selection import train_test_split
from src.utils.helper import load_config

DATA_SRC_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths and split sizes."""

    raw_data_path: str = Path("data") / "raw" / "telco-churn.csv"
    train_data_path: str = Path("data") / "raw" / "train.csv"
    val_data_path: str = Path("data") / "raw" / "val.csv"
    test_data_path: str = Path("data") / "raw" / "test.csv"
    val_size: int | float = None
    test_size: int | float = None
    random_state: int = None

    def __post_init__(self):
        config = load_config()
        if self.val_size is None:
            self.val_size = config["data"]["val_size"]
        if self.test_size is None:
            self.test_size = config["data"]["test_size"]
        if self.random_state is None:
            self.random_state = config["data"]["random_state"]


class DataIngestion:
    """Ingest dataset and produce training, validation, and test splits."""

    def __init__(self, config: DataIngestionConfig = None):
        """Initialize the ingestion workflow with a configuration object."""
        if config is None:
            self.ingestion_config = DataIngestionConfig()
        else:
            self.ingestion_config = config

    def initiate_data_ingestion(self):
        """Download the dataset, save raw CSV, and create train/val/test splits.

        Returns:
            tuple[str, str, str]: Paths to the train, validation, and test CSV files.
        """
        logger.info("Initiated data ingestion")
        try:
            df = pd.read_csv(DATA_SRC_URL)
            logger.info("Read the dataset as Pandas dataframe")
            raw_data_path = Path(self.ingestion_config.raw_data_path).parent
            raw_data_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, header=True)

            logger.info("Train Test split initiated")
            temp_set, test_set = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
                stratify=df["Churn"],
            )
            train_set, val_set = train_test_split(
                temp_set,
                test_size=self.ingestion_config.val_size,
                random_state=self.ingestion_config.random_state,
                stratify=temp_set["Churn"],
            )
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            val_set.to_csv(
                self.ingestion_config.val_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logger.info("Data Ingestion Completed. Files written to data/raw/*.csv")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
