# import steps pipeline
from steps.data_ingest_step import data_ingest_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step

from steps.outlier_detection_step import outlier_detection_step
from steps.data_spliter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evalution_step import model_evaluation_step


# pipine and model tracking
import mlflow
from zenml import pipeline


# for generate dummy data for example
from utils.generate_data import generate_data


@pipeline(
    enable_cache=False,
)
def training_pipeline():
    data = generate_data()

    # mlflow running tracking experiment
    with mlflow.start_run():

        # standardize features
        data = feature_engineering_step(
            df=data,
            features=["feature_1", "feature_2", "feature_3", "feature_4"],
            target=["target_ABC"],
            method="normalize",
        )

        # split data
        data = data_splitter_step(
            method="train-test",
            df=data,
            features=["feature_1", "feature_2", "feature_3", "feature_4"],
            target=["target_ABC"],
            test_size=0.2,
        )

        # model building
        trained_model = model_building_step(
            df=data,
            type_task="classification",
            model_name="decision_tree",
        )

        # # evaluation
        model_evaluation_step(
            df=data,
            trained_model=trained_model,
            task_type="classification",
        )


if __name__ == "__main__":
    # data ingest step
    # handling missing value step
    # feature engineering step
    # outlier detection step
    # data splitting step
    # model building step
    # model evaluation step
    pass
