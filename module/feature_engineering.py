from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
import pandas as pd


import logging
import mlflow


# standardrization
def normalize_data(
    df: pd.DataFrame, features: list[str], target: list[str], method: str = "standard"
) -> pd.DataFrame:
    """
    Normalizes numeric features in a DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    features (list): List of numeric columns to be normalized.
    method (str, default='standard'): Normalization method ('standard' for Z-Score, 'minmax' for Min-Max).

    Returns:
    pd.DataFrame: DataFrame with normalized features.
    """

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'.")

    # Normalisasi kolom-kolom yang ditentukan
    df[features] = scaler.fit_transform(X=df[features], y=df[target])

    # logging.info(f"Successfully normalize data using {method}")
    mlflow.log_param("Normalize data", f"Successfully normalize data using {method}")
    return df


# Categorical encoding
def label_encoder(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Encodes categorical features in a DataFrame with Label Encoder.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    columns (list): List of categorical columns to encode.

    Returns:
    pd.DataFrame: DataFrame with encoded columns.
    """
    le = LabelEncoder()

    for col in columns:
        df[col] = le.fit_transform(df[col])

    # logging.info("Successfully Encode Categorical Features")
    mlflow.log_param("Label Encoder data", "Successfully Encode Categorical Features")
    return df


def one_hot_encoder(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:

    ohe = OneHotEncoder()

    df[features] = ohe.fit_transform(X=df[features])

    # logging.info("Successfully Encode Categorical Features")
    mlflow.log_param("One Hot Encoder data", "Successfully Encode Categorical Features")
    return df


# Custom functions
def create_feature_interaction(
    df: pd.DataFrame, columns: list, interaction_name: str
) -> pd.DataFrame:
    """
    Creates an interaction feature from two or more numeric features in a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): A list of columns for the interaction feature.
    interaction_name (str): The name of the new interaction feature.

    Returns:
    pd.DataFrame: The DataFrame with the new interaction feature.
    """

    # example of interaction - just custom it!
    df[interaction_name] = (
        df[columns[0]] * df[columns[1]]
    )  # Misalnya perkalian dua fitur numerik

    # logging.info("Successfully create Feature with name: {}".format(interaction_name))
    mlflow.log_param(
        "Create Interaction",
        "Successfully create Feature with name: {}".format(interaction_name),
    )
    return df


if __name__ == "__main__":
    pass
