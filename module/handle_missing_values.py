import pandas as pd
import logging
from sklearn.impute import SimpleImputer


def impute_missing_values(
    df: pd.DataFrame, strategy: str = "mean", columns: list = []
) -> pd.DataFrame:
    """Impute missing values for the given columns.

    Args:
        df (pd.DataFrame): dataFrame to impute missing values.
        strategy (str, optional): Strategy to fill data (most_frequent, mean, median, constant). Defaults to "mean".
        columns (list, optional): What the column want to impute. Defaults to [].

    Returns:
        pd.DataFrame: DataFrame after impute missing values.
    """

    if not columns:
        columns = (
            df.columns.tolist()
        )  # Jika tidak ada kolom yang diberikan, imputasi untuk semua kolom

    imputer = SimpleImputer(strategy=strategy)

    # Melakukan imputasi untuk kolom-kolom yang ditentukan
    df[columns] = imputer.fit_transform(df[columns])

    logging.info("Successfully Impute missing values with {} strategy".format(strategy))
    return df


def drop_missing_values(
    df: pd.DataFrame, axis: int = 0, threshold: float = 0.5
) -> pd.DataFrame:
    """drop missing values from DataFrame by expecting value bigger or same than thereshold.

    Args:
        df (pd.DataFrame): DataFrame.
        axis (int, optional): 0 for the rows and 1 for the columns. Defaults to 0.
        threshold (float, optional): thereshold of values. Defaults to 0.5.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    if axis == 0:
        # Hapus baris dengan missing values lebih dari threshold
        logging.info("Successfully drop missing values in rows")
        return df.dropna(axis=0, thresh=int(threshold * len(df.columns)))
    elif axis == 1:
        # Hapus kolom dengan missing values lebih dari threshold
        logging.info("Successfully drop missing values in columns")
        return df.dropna(axis=1, thresh=int(threshold * len(df)))
    else:
        raise ValueError("Axis must be 0 (for rows) or 1 (for columns).")


def fill_missing_values(
    df: pd.DataFrame, value: float, columns: list = []
) -> pd.DataFrame:
    """Fill missing values with specified value.

    Args:
        df (pd.DataFrame): dataFrame.
        value (float): fill value.
        columns (list, optional): columns to fill. Defaults to [].

    Returns:
        pd.DataFrame: _description_
    """
    if not columns:
        columns = (
            df.columns.tolist()
        )  # Jika tidak ada kolom yang diberikan, isi semua kolom

    # Mengisi missing values pada kolom yang ditentukan dengan nilai yang diberikan
    df[columns] = df[columns].fillna(value)

    logging.info("Successfully fill missing values with value: {}".format(value))
    return df


if __name__ == "__main__":
    pass
