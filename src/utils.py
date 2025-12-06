import numpy as np
import pandas as pd
import dill
import os
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException


# ---------------------------------------------------------
# Save any Python object using dill
# ---------------------------------------------------------
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)


# ---------------------------------------------------------
# Generic Model Evaluation Function
# ---------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train a model, make predictions, and compute metrics.
    Works for ANY sklearn-compatible model.
    """
    try:
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        return {
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "mae": mean_absolute_error(y_test, y_pred_test),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        }

    except Exception as e:
        raise CustomException(e, sys)
