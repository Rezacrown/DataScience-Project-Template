from module.data_splitter import split_data, split_data_kfold
import pandas as pd


from zenml import step
from typing_extensions import Annotated


@step
def data_splitter_step(
    df: pd.DataFrame,
    features: list[str],
    target: list[str],
    method: str = "train-test",
    n_splits: int = 5,
    test_size: float = 0.2,
    stratify: bool = False,
) -> dict:
    """
    Function to divide data based on the selected method, whether Train-Test Split or K-Fold Cross Validation.

    Args:
        df (pd.DataFrame): Data to be divided.
        features (list): features (X) to be used.
        target (str): target (y) to be used.
        method (str, default='train-test'): Data division method ('train-test' or 'kfold').
        n_splits (int, default=5): Number of folds for K-Fold Cross Validation.
        test_size (float, default=0.2): Percentage of data used as test in the Train-Test Split method.
        stratify (bool, default=False): Balanced data division based on labels.

    Returns:
        list or tuple: If the method is 'train-test', returns a tuple containing the train and test DataFrame.
        If the method is 'kfold', returns a list containing the train and test tuples for each fold.
    """

    if method == "train-test":
        # Gunakan metode Train-Test Split
        data_spliting = split_data(
            df=df,
            test_size=test_size,
            stratify=stratify,
            features=features,
            target=target,
            random_state=42,
        )

        return data_spliting

    elif method == "kfold":
        # Gunakan metode K-Fold Cross Validation
        return split_data_kfold(
            df=df, n_splits=n_splits, features=features, target=target, random_state=42
        )

    else:
        raise ValueError("Method must be 'train-test' or 'kfold'.")


if __name__ == "__main__":
    pass
