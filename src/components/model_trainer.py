import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    artifacts_dir: Path = Path("artifacts")
    trained_model_file: Path = artifacts_dir / "model.pkl"


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(self.config.artifacts_dir, exist_ok=True)

    def initiate_model_training(self, train_array: np.ndarray, test_array: np.ndarray):
        try:
            logging.info("Splitting train and test arrays...")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info("Initializing candidate models...")
            models = {
                "RandomForestRegressor": RandomForestRegressor(n_estimators=120, random_state=42),
                "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                "XGBRegressor": XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                ),
                "LinearRegression": LinearRegression(),
            }

            best_model_name = None
            best_score = -np.inf
            best_model = None
            results = {}

            for name, model in models.items():
                logging.info(f"Training model: {name}")

                metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
                results[name] = metrics

                logging.info(f"{name} Test R2 Score: {metrics['test_r2']}")

                # Select best model
                if metrics["test_r2"] > best_score:
                    best_score = metrics["test_r2"]
                    best_model_name = name
                    best_model = model

            logging.info(f"Best Model Selected → {best_model_name} (R2: {best_score})")

            # Save best model
            save_object(
                file_path=self.config.trained_model_file,
                obj=best_model
            )

            logging.info(f"Model saved at: {self.config.trained_model_file}")

            return {
                "best_model_name": best_model_name,
                "best_r2_score": best_score,
                "all_results": results,
                "saved_model_path": self.config.trained_model_file,
            }

        except Exception as e:
            raise CustomException(e, sys)
