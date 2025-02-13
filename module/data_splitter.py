from typing import Tuple
from typing_extensions import Annotated

from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np

import logging
import mlflow


def split_data(
    df: pd.DataFrame,
    features: list[str],
    target: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[
    # Notice we use a Tuple and Annotated to return
    # multiple named outputs
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Splitting the dataset into train and test data.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        features (list): the features (X) to be used.
        target (str): the target (y) to be used.
        test_size (float, default=0.2): The percentage of data to be used as test data.
        random_state (int, default=42): Seed for randomizing the data.
        stratify (bool, default=False): Whether the data is split evenly based on labels.

    Returns:
        tuple: Returns a tuple containing 4 DataFrames, X_train, X_test, y_train and y_test.

    """

    # pisahkan X Feature dan y target
    if len(features) > 0:
        X = df[features]  # select features yang akan digunakan jika ada
    else:
        X = df.drop(target)

    y = df[target]

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # logging.info("Successfully Split Data Training and Data Testing")
    mlflow.log_param("Split data", "Successfully Split Data Training and Data Testing")
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def split_data_kfold(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> list:
    """
    Splitting the dataset into folds for K-Fold Cross Validation.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        features (list): The features (X) to be used.
        target (str): The target (y) to be used.
        n_splits (int, default=5): The desired number of folds.
        random_state (int, default=42): Seed for randomizing the data.

    Returns:
        list: A list of tuples containing the train and test DataFrames for each fold.
    """

    if len(features) > 0:
        X = df[features]
    else:
        X = df.drop(columns="target")  # Misalkan kolom target adalah kolom label

    y = df[target]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Gabungkan kembali X_train dan y_train menjadi satu DataFrame untuk train dan test
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        splits.append((train_df, test_df))

    # logging.info("Successfully Split Data Training and Data Testing")
    mlflow.log_param("Split data", "Successfully Split Data Training and Data Testing")

    return splits


if __name__ == "__main__":
    pass
