import numpy as np
import pandas as pd

import logging


def z_score_outlier_detection(
    df: pd.DataFrame, columns: list, threshold: float = 3.0
) -> pd.DataFrame:
    """
    Outlier detection using Z-Score.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): List of columns to be analyzed for outliers.
        threshold (float, default=3.0): Z-Score threshold for detecting outliers (generally more than 3 is considered an outlier).

    Returns:
        pd.DataFrame: DataFrame with columns that are already clean from outliers.
    """
    z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())

    logging.info(
        f"Successfully Clean DataFrames from outliers using Z Score Strategy with Threshold {threshold}:.2f"
    )
    return df[(z_scores < threshold).all(axis=1)]


def iqr_outlier_detection(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Outlier detection using IQR (Interquartile Range).

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): List of columns to be analyzed for outliers.

    Returns:
        pd.DataFrame: DataFrame with columns that are already clean from outliers.
    """
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # return data yang tidak termasuk outlier pada kolom
    logging.info("Successfully Clean DataFrames from outliers using IQR Strategy")
    return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]


def quantile_outlier_detection(
    df: pd.DataFrame,
    columns: list,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> pd.DataFrame:
    """
    Outlier detection using quantiles.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): List of columns to be analyzed for outliers.
        lower_quantile (float, default=0.05): Lower quantile for outlier detection.
        upper_quantile (float, default=0.95): Upper quantile for outlier detection.

    Returns:
        pd.DataFrame: DataFrame with columns cleaned of outliers.
    """
    lower_bound = df[columns].quantile(lower_quantile)
    upper_bound = df[columns].quantile(upper_quantile)

    logging.info(
        f"Successfully Clean DataFrames from outliers using Quantile Strategy with Lower Quantile{lower_quantile} and Upper Quantile {upper_quantile}"
    )
    return df[(df[columns] >= lower_bound) & (df[columns] <= upper_bound)].dropna()


if __name__ == "__main__":
    pass
