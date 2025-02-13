from module.model_evaluation import evaluate_classification, evaluate_regression

from typing import Any

from zenml import step


@step
def model_evaluation_step(
    df: dict,
    task_type="classification",
    trained_model=None,
) -> Any:
    """
    Evaluasi model berdasarkan jenis tugas (classification/regression).

    Args:
        y_test (pd.Series): Label sebenarnya.
        y_pred (pd.Series): Prediksi model.
        task_type (str, default="classification"): Jenis tugas ("classification" atau "regression").

    Returns:
        dict: Hasil evaluasi sesuai dengan jenis model.
    """
    # check training_model has inputed
    if trained_model is None:
        raise ValueError("Please Input the trained model to Predict")

    # split data test
    print("Split test data starting")
    X_test = df["X_test"]
    y_test = df["y_test"]

    # predict model
    print("Predicting model starting")
    y_pred = trained_model.predict(X_test)

    if task_type == "classification":
        return evaluate_classification(y_test, y_pred)
    elif task_type == "regression":
        return evaluate_regression(y_test, y_pred)
    else:
        raise ValueError("task_type harus 'classification' atau 'regression'")


if __name__ == "__main__":
    pass
