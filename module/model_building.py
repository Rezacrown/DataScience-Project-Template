import pandas as pd
import numpy as np

# model import
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

# import xgboost as xgb
from sklearn.base import BaseEstimator
from typing import Dict, Any

import logging
import mlflow


def build_model(
    type_task: str, model_name: str, hyperparams: Dict[str, Any] = None
) -> BaseEstimator:
    """
    Builds a model based on a given name.

    Parameters:
        type_task (str): regression or classification
        model_name (str): The type of model to use ('linear_regression', 'decision_tree', 'random_forest', 'svm').
        hyperparams (dict, optional): Hyperparameters to use in the model.

    Returns:
        model: The machine learning model that has been created.
    """

    if type_task == "regression":
        models = {
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(),
            "random_forest": RandomForestRegressor(),
            "svm": SVR(),
        }
    elif type_task == "classification":
        models = {
            "logistic_regression": LogisticRegression(),
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "svm": SVC(),
        }
    else:
        raise ValueError("type model is between regression and classification")

    if model_name not in models:
        raise ValueError(
            f"Model '{model_name}' not supported. choice from {list(models.keys())}"
        )

    model = models[model_name]

    if hyperparams:
        model.set_params(**hyperparams)

    # logging.info(f"Successfully Build {model_name} Model Architecture of {type_task}")
    mlflow.log_param(
        "Build Model Architecture",
        f"Successfully Build {model_name} Model Architecture of {type_task}",
    )
    return model


def train_model(model, X_train, y_train) -> BaseEstimator:
    """
    Train a model with training data.

    Parameters:
        model: The machine learning model that has been created.
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Target training data.

    Returns:
        model: The trained model.
    """
    model.fit(X_train, y_train)

    # logging.info("Successfully Train Model")
    mlflow.log_param("Training Model", "Successfully Train Model")
    return model


def tune_hyperparameters(
    model, X_train, y_train, param_grid: Dict[str, list], cv: int = 5
):
    """
    Perform hyperparameter tuning with GridSearchCV.

    Parameters:
        model: The model to be tuned.
        x_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The target training data.
        param_grid (dict): The hyperparameter grid to try.
        cv (int): The number of cross-validations (default 5).

    Returns:
        best_model: The model with the best hyperparameters.
        best_params: The best hyperparameters found.
    """
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring="r2")
    grid_search.fit(X_train, y_train)

    # logging.info("Successfully Tuning Hyperparameter of Model")
    mlflow.log_param(
        "Tunning Hyperparameter of Model", "Successfully Tuning Hyperparameter of Model"
    )
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model with test data.

    Parameters:
        model: The trained model.
        X_test (pd.DataFrame): Test data features.
        y_test (pd.Series): Test data targets.

    Returns:
        score (float): Model evaluation score.
    """

    # logging.info("Successfully Evaluating Model")
    mlflow.log_param("Evaluate Model Score", "Successfully Evaluating Model")
    return model.score(X_test, y_test)


if __name__ == "__main__":
    pass
