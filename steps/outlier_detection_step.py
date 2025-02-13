import pandas as pd

from module.outlier_detection import (
    z_score_outlier_detection,
    iqr_outlier_detection,
    quantile_outlier_detection,
)


from zenml import step


@step
def outlier_detection_step(
    df: pd.DataFrame,
    method: str = "z_score",
    columns: list = [],
    threshold: float = 3.0,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> pd.DataFrame:
    """
    Outlier detection steps based on the selected method.

    Args:
        df (pd.DataFrame): DataFrame to be processed.
        method (str, default='z_score'): Outlier detection method used ('z_score', 'iqr', 'quantile').
        columns (list): Columns to be checked for outliers.
        threshold (float, default=3.0): Z-Score threshold for the 'z_score' method.
        lower_quantile (float, default=0.05): Lower quantile for the 'quantile' method.
        upper_quantile (float, default=0.95): Upper quantile for the 'quantile' method.

    Returns:
        pd.DataFrame: DataFrame without outliers according to the selected method.

    Examples:
    ### Suppose we have a DataFrame df
    >>> df = pd.read_csv('path/to/your/data.csv')

    ### Outlier detection using Z-Score on 'age' and 'salary' columns
    >>> df_no_outliers_zscore = outlier_detection_step(df, method='z_score', columns=['age', 'salary'], threshold=3.0)

    ### Outlier detection using IQR on 'age' and 'salary' columns
    >>> df_no_outliers_iqr = outlier_detection_step(df, method='iqr', columns=['age', 'salary'])

    ### Outlier detection using quantiles on 'age' and 'salary' columns
    >>> df_no_outliers_quantile = outlier_detection_step(df, method='quantile', columns=['age', 'salary'], lower_quantile=0.05, upper_quantile=0.95)

    """
    if method == "z_score":
        # Deteksi outlier menggunakan Z-Score
        return z_score_outlier_detection(df, columns, threshold)

    elif method == "iqr":
        # Deteksi outlier menggunakan IQR
        return iqr_outlier_detection(df, columns)

    elif method == "quantile":
        # Deteksi outlier menggunakan kuantil
        return quantile_outlier_detection(df, columns, lower_quantile, upper_quantile)

    else:
        raise ValueError("Method must be 'z_score', 'iqr', or 'quantile'.")


if __name__ == "__main__":
    pass
