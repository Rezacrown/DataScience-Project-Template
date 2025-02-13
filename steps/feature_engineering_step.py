import pandas as pd

from module.feature_engineering import (
    normalize_data,
    label_encoder,
    create_feature_interaction,
)

from typing_extensions import Annotated
from zenml import step


@step
def feature_engineering_step(
    df: pd.DataFrame,
    method: str = "normalize",
    features: list[str] = [],
    target: list[str] = [],
    interaction_columns=[],
    interaction_name: str = "",
) -> pd.DataFrame:
    """
    Feature processing steps based on the selected method.

    Args:
        df (pd.DataFrame): DataFrame to be processed.
        method (str, default='normalize'): Feature processing method to be applied ('normalize', 'encode', 'interaction').
        columns (list, default=[]): Columns involved in a particular method.
        interaction_columns (list, default=[]): Columns to create interaction features.
        interaction_name (str, default=''): Name of the new interaction feature.

    Returns:
        pd.DataFrame: DataFrame that has been processed.

    examples:
        ### Suppose we have a DataFrame df
        >>> df = pd.read_csv('path/to/your/data.csv')

        ### Normalization of numeric features
        >>> df_normalized = feature_engineering_step(df, method='normalize', columns=['age', 'salary'])

        ### Categorical feature encoding
        >>> df_encoded = feature_engineering_step(df, method='encode', columns=['gender', 'city'])

        ### Create interaction features
        >>> df_interaction = feature_engineering_step(df, method='interaction', interaction_columns=['age', 'salary'], interaction_name='age_salary_interaction')
    """
    if method == "normalize":
        # Melakukan normalisasi pada kolom-kolom yang ditentukan
        df = normalize_data(df=df, features=features, target=target, method="standard")

        return df

    elif method == "encode":
        # Melakukan encoding pada kolom-kolom kategorikal
        return label_encoder(df=df, columns=features)

    elif method == "interaction":
        # Membuat fitur interaksi dari dua kolom numerik
        return create_feature_interaction(df, interaction_columns, interaction_name)

    else:
        raise ValueError("Method must be 'normalize', 'encode', or 'interaction'.")
