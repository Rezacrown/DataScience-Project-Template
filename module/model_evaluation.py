import pandas as pd
import logging
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


from typing import Union


def evaluate_classification(
    y_true: Union[pd.DataFrame, pd.Series], y_pred: Union[pd.DataFrame, pd.Series]
) -> dict:
    """
    Evaluate a classification model using standard metrics.

    Args:
        y_true: True labels.
        y_pred: Model predictions.

    Returns:
        dict: Model evaluation results (accuracy, precision, recall, F1-score).
    """

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }

    # logging.info(f"Evaluating Model Successfully")
    mlflow.log_param("Model Evaluation", "Evaluating Model Successfully")
    # logging.info(result)
    mlflow.log_metric("Accuracy", result["accuracy"])
    mlflow.log_metric("precision", result["precision"])
    mlflow.log_metric("recall", result["recall"])
    mlflow.log_metric("f1_score", result["f1_score"])

    return result


def evaluate_regression(
    y_true: Union[pd.DataFrame, pd.Series], y_pred: Union[pd.DataFrame, pd.Series]
) -> dict:
    """
    Evaluate the regression model using standard metrics.

    Args:
        y_true: True labels.
        y_pred: Model predictions.

    Returns:
        dict: Model evaluation results (MAE, MSE, RMSE, R2-score).
    """

    result = pd.DataFrame(
        {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
        }
    )

    # logging.info(f"Evaluating Model Successfully")
    mlflow.log_param("Model Evaluation", "Evaluating Model Successfully")
    # logging.info(result)
    mlflow.log_metric("mae", result["mae"])
    mlflow.log_metric("mse", result["mse"])
    mlflow.log_metric("rmse", result["rmse"])
    mlflow.log_metric("r2_score", result["r2_score"])
    return result


if __name__ == "__main__":
    pass
