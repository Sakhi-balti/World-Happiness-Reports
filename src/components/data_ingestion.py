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

# -----------------------------
# Script execution (for testing)
# -----------------------------
if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformations
        from src.components.model_trainer import ModelTrainer

        # -------------------------------
        # 1) DATA INGESTION
        # -------------------------------
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # -------------------------------
        # 2) DATA TRANSFORMATION
        # -------------------------------
        transformer = DataTransformations()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            train_path, test_path
        )

        # -------------------------------
        # 3) MODEL TRAINING
        # -------------------------------
        trainer = ModelTrainer()
        results = trainer.initiate_model_training(train_arr, test_arr)

        # -------------------------------
        # PRINT SUMMARY
        # -------------------------------
        print("\n============== TRAINING COMPLETED ==============")
        print(f"Best Model       : {results['best_model_name']}")
        print(f"Best Test R²     : {results['best_r2_score']}")
        print(f"Model Saved At   : {results['saved_model_path']}")
        print("================================================\n")

    except Exception as e:
        print("Pipeline Failed:", e)
