import numpy as np
import pandas as pd

from zenml import step


@step
def generate_data():
    # Define column names
    feature_columns = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
    label_column = ["target_ABC"]

    # Generate random data
    np.random.seed(42)
    data = np.random.rand(20, 5)  # 20 rows, 5 features
    labels = np.random.choice(
        ["A", "B", "C"], size=20
    )  # 20 rows, 1 label with 3 possible values

    # Create DataFrame
    df = pd.DataFrame(
        np.hstack((data, labels.reshape(-1, 1))), columns=feature_columns + label_column
    )

    return df
