import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

from src.components.data_transformation import  DataTransformations
@dataclass
class DataIngestionConfig:
    artifacts_dir: Path = Path("artifacts")
    train_data_path: Path = artifacts_dir / "train.csv"
    test_data_path: Path = artifacts_dir / "test.csv"
    raw_data_path: Path = artifacts_dir / "raw.csv"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            # Read dataset
            df = pd.read_csv(r"data/processed/world_happiness_cleaned.csv")
            logging.info(f"Dataset Loaded with shape: {df.shape}")

            # Create directories
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # Split data
            logging.info("Train-test split started")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train/test files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return (
                str(self.ingestion_config.train_data_path),
                str(self.ingestion_config.test_data_path),
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformations()
    data_transformation.initiate_data_transformation(train_data, test_data)