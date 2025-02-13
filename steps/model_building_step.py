from module.model_building import (
    build_model,
    train_model,
    tune_hyperparameters,
)

from typing_extensions import Annotated
from typing import Union
from sklearn.base import (
    RegressorMixin,
    ClassifierMixin,
)

from zenml import step, Model
from zenml.client import Client

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker


# create Model instances for MLflow experiment tracking
model = Model(
    name="example_model_predictions",
)


@step(experiment_tracker=experiment_tracker, model=model)
def model_building_step(
    df: dict,
    model_name: str,
    type_task: str = "regression",
    tuning=False,
    hyperparams=None,
    param_grid=None,
) -> Annotated[
    Union[RegressorMixin, ClassifierMixin],
    "Return Model after Training with Train dataset",
]:
    """
    Steps to build, train, and evaluate a model.

    Parameters:
        df (pd.DataFrame): The dataset used.
        X_train: X_train data.
        y_train: y_train data.
        type_task: The type of task used (regression or classification).
        model_name (str): The type of model ('linear_regression', 'decision_tree', 'random_forest', 'svm', 'logistic_regression').
        tuning (bool): Whether to perform hyperparameter tuning (default False).
        hyperparams (dict, optional): Hyperparameters for the model (default None).
        param_grid (dict, optional): Hyperparameter grid if tuning is enabled.

    Returns:
        trained_model: The trained model.
        score: The model evaluation score.
        best_params (optional): The best hyperparameters if tuning is enabled.
    """

    # split dataset for training
    X_train = df["X_train"]
    y_train = df["y_train"]

    # Builld Model Architecture
    model = build_model(
        type_task=type_task,
        model_name=model_name,
        hyperparams=hyperparams,
    )

    # tunning with GridSearch if True and else for None
    if tuning and param_grid:
        trained_model_tunning, best_params = tune_hyperparameters(
            model, X_train, y_train, param_grid
        )
    else:
        best_params = None

    # train model with data training
    trained_model = train_model(model, X_train, y_train)

    # return Model
    if tuning:
        return trained_model_tunning, best_params
    else:
        return trained_model


if __name__ == "__main__":
    pass
