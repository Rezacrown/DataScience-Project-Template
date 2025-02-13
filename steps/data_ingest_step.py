import pandas as pd
from typing_extensions import Annotated

from module.data_ingest import ingest_file, ingest_zip_file
from types import UnionType

from zenml import step


@step
def data_ingest_step(
    file_path: str,
) -> pd.DataFrame:
    """Step to ingest data from raw data files like (.csv, .zip, .json) to pandas Dataframe.

    Args:
        file_path (str): path where the data stored.

    Returns:
        pandas DataFrame or Pandas Series.
    """

    if file_path.endswith(".zip"):

        df = ingest_zip_file(file_path)
        return df
    else:
        df = ingest_file(file_path)
        return df


if __name__ == "__main__":
    pass
