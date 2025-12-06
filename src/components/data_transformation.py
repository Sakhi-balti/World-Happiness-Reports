from typing import Tuple
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    artifacts_dir: Path = Path("artifacts")
    preprocessor_obj_file: Path = artifacts_dir / "preprocessor.pkl"


class DataTransformations:
    def __init__(self):
        self.config = DataTransformationConfig()
        os.makedirs(self.config.artifacts_dir, exist_ok=True)

    def _build_preprocessor(self) -> ColumnTransformer:
        """Creates preprocessing pipeline"""
        try:
            num_columns = [
                "Year", "Corruption", "GDP per Capita", "SocialSupport",
                "LifeExpectancy", "Freedom", "Generosity"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, Path]:

        try:
            logging.info("Loading train and test datasets")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = "HappinessScore"
            preprocessor = self._build_preprocessor()

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data Transformation Completed")

            train_array = np.c_[X_train_transformed, y_train.values]
            test_array = np.c_[X_test_transformed, y_test.values]

            save_object(
                file_path=self.config.preprocessor_obj_file,
                obj=preprocessor
            )

            return train_array, test_array, self.config.preprocessor_obj_file

        except Exception as e:
            raise CustomException(e, sys)
