import pandas as pd

from module.handle_missing_values import (
    impute_missing_values,
    drop_missing_values,
    fill_missing_values,
)


from zenml import step


@step
def handle_missing_values_step(
    df: pd.DataFrame,
    method: str = "impute",
    strategy: str = "mean",
    columns: list = [],
    value: float = None,
    axis: int = 0,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Missing values ​​processing steps based on the selected method.

    Args:
        df (pd.DataFrame): DataFrame to process.
        method (str, default='impute'): Method for handling missing values ​​('impute', 'drop', 'fill').
        strategy (str, default='mean'): Imputation strategy for missing values ​​('mean', 'median', 'most_frequent').
        columns (list, default=[]): Columns to be processed.
        value (float, default=None): Value to fill in missing values ​​if using the 'fill' method.
        axis (int, default=0): If using the 'drop' method, specify whether to delete rows (0) or columns (1).
        threshold (float, default=0.5): The threshold for the proportion of missing values ​​that allows rows/columns to be maintained.

    Returns:
        pd.DataFrame: Processed DataFrame.

    Examples:
        ### Suppose we have a DataFrame df
        >>> df = pd.read_csv('path/to/your/data.csv')

        ### Imputation of missing values ​​with the average
        >>> df_imputed = handle_missing_values_step(df, method='impute', strategy='mean', columns=['age', 'salary'])

        ### Delete rows with more than 50% missing values
        >>> df_dropped = handle_missing_values_step(df, method='drop', axis=0, threshold=0.5)

        ### Fill in missing values ​​with a specific value (for example 0)
        >>> df_filled = handle_missing_values_step(df, method='fill', value=0, columns=['age', 'salary'])

    """
    if method == "impute":
        # Imputasi missing values
        return impute_missing_values(df, strategy, columns)

    elif method == "drop":
        # Menghapus baris atau kolom dengan missing values
        return drop_missing_values(df, axis, threshold)

    elif method == "fill":
        # Mengisi missing values dengan nilai tertentu
        return fill_missing_values(df, value, columns)

    else:
        raise ValueError("Method must be 'impute', 'drop', or 'fill'.")


if __name__ == "__main__":
    pass
