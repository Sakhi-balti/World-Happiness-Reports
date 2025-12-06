
from typing import Tuple, List
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

    # ---------------------------------------------------------
    # Drop columns with too much missing data
    # ---------------------------------------------------------
    def _find_bad_columns(self, df: pd.DataFrame, threshold: float = 0.4) -> List[str]:
        missing_ratio = df.isna().mean()
        bad_cols = missing_ratio[missing_ratio > threshold].index.tolist()

        if bad_cols:
            logging.warning(f"Dropping high-missing columns: {bad_cols}")

        return bad_cols

    # ---------------------------------------------------------
    # Build preprocessing pipeline
    # ---------------------------------------------------------
    def _build_preprocessor(
        self, numeric_columns: List[str], categorical_columns: List[str]
    ) -> ColumnTransformer:

        try:
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            # Full preprocessing
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ],
                remainder="drop"
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------------------------------------
    # Main transformation method
    # ---------------------------------------------------------
    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, Path]:

        try:
            logging.info("Loading train and test datasets...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = "HappinessScore"

            # Drop rows with missing target
            train_df = train_df.dropna(subset=[target_col])
            test_df = test_df.dropna(subset=[target_col])

            # Remove bad columns
            bad_cols = list(
                set(self._find_bad_columns(train_df) + self._find_bad_columns(test_df))
            )

            if bad_cols:
                train_df = train_df.drop(columns=bad_cols)
                test_df = test_df.drop(columns=bad_cols)

            # ---------------------------------------------------------
            # Auto detect numerical & categorical features
            # ---------------------------------------------------------
            numeric_columns = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_columns = train_df.select_dtypes(include=["object"]).columns.tolist()

            # Remove target from auto-detected lists
            if target_col in numeric_columns:
                numeric_columns.remove(target_col)
            if target_col in categorical_columns:
                categorical_columns.remove(target_col)

            logging.info(f"Numeric Columns: {numeric_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")

            # Train/test split
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            # Build and apply preprocessor
            preprocessor = self._build_preprocessor(
                numeric_columns, categorical_columns
            )

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data Transformation Completed Successfully.")

            # Combine transformed features + target
            train_array = np.c_[X_train_transformed, y_train.values]
            test_array = np.c_[X_test_transformed, y_test.values]

            # Save preprocessor
            save_object(
                file_path=self.config.preprocessor_obj_file,
                obj=preprocessor
            )

            return (
                train_array,
                test_array,
                self.config.preprocessor_obj_file
            )

        except Exception as e:
            raise CustomException(e, sys)
