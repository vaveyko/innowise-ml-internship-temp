from typing import Literal, Any, Callable, Iterable

from pandas import DataFrame, read_csv
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator


class ModelValidation:
    def __init__(
        self,
        data: DataFrame,
        y: str,
        x: str | list[str] = None,
    ):
        """
        :param data:
        :param y: target column name
        :param x: features columns names or all the rest except y
        """
        self.__y_data = data[y]
        self.__x_data = data.drop(y, axis=1) if x is None else data[x]
        self.__metrics: dict[str, list[float]] = {}
        self.__metrics_func: list[Callable[[Any, Any], float]] = []

    def split_data(
        self,
        n_splits: int = 5,
        window_type: Literal["expanding", "sliding"] = "sliding",
        train_size: int = 1,
    ):
        """
        The function transforms a dataset into a series of "train-test".
        It supports two validation strategies -- Expanding and Sliding controlled by the
        window_type parameter.

        :param n_splits:
        :param window_type:
        :param train_size: size of train subset in folds,
        train_size mast be lower than n_split (used only if window_type is "sliding")
        :return: generator of X_train, X_test, y_train, y_test
        """
        splitter: TimeSeriesSplit
        match window_type:
            case "expanding":
                splitter = TimeSeriesSplit(n_splits=n_splits)
            case "sliding":
                # as in documentation by default max size of test subset is
                # n_samples // (n_splits + 1)
                # so I use this as a size of one fold
                # set max train size as n_fold * fold_size
                max_train_size = train_size * len(self.__x_data) // (n_splits + 1)

                splitter = TimeSeriesSplit(
                    n_splits=n_splits,
                    max_train_size=max_train_size,
                )

        for i, (train_idx, test_idx) in enumerate(splitter.split(self.__x_data)):
            yield (
                self.__x_data.iloc[train_idx],
                self.__x_data.iloc[test_idx],
                self.__y_data.iloc[train_idx],
                self.__y_data.iloc[test_idx],
            )

    def init_metrics(
        self, metrics_functions: list[Callable[[Any, Any], float]]
    ) -> None:
        """
        Initialize metric functions and prepare storage for their results.
        :param metrics_functions:
        :return:
        """
        self.__metrics = {}
        self.__metrics_func = metrics_functions
        for metric in metrics_functions:
            self.__metrics[metric.__name__] = []

    def calculate_metric(
        self, y_pred: Any, y_true: Any, return_value: bool = False
    ) -> None | list[tuple[str, float]]:
        """
        Calculate given metrics for current predictions

        :param y_pred:
        :param y_true:
        :param return_value:
        :return: list of results [(metric_name, value)] if return_value is True
        """
        results = [] if return_value else None
        for metric_func in self.__metrics_func:
            metric_name = getattr(metric_func, "__name__", repr(metric_func))
            metric_value = metric_func(y_pred, y_true)
            self.__metrics[metric_name].append(metric_value)

            if results is not None:
                results.append((metric_name, metric_value))
        return results

    def calculate_metrics_mean(self) -> list[tuple[str, float]]:
        """
        Calculate average meaning of train metrics
        :return: [(metric_name, metric_mean), (...), ...]
        """
        return [
            (name, float(np.mean(all_metrics)))
            for name, all_metrics in self.__metrics.items()
        ]


# ========= Metrics functions =========


def MSE(y_pred: Any, y_true: Any) -> float:
    return np.mean((y_pred - y_true) ** 2)


def RMSE(y_pred: Any, y_true: Any) -> float:
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def MAE(y_pred: Any, y_true: Any) -> float:
    return np.mean(np.abs(y_pred - y_true))


def fit_validate(
    model: BaseEstimator,
    splitter: Iterable,
    metrics_func: list[Callable[[Any, Any], float]],
    validation_model: ModelValidation,
    show_logs: bool = False,
):
    """
    Fit model on each split of data and validate on test
    model parameter must support fit, predict methods
    """
    validation_model.init_metrics(metrics_func)

    for i, (X_train, X_test, y_train, y_test) in enumerate(splitter, start=1):
        model.fit(X_train, y_train)
        y_prediction = model.predict(X_test)
        results = validation_model.calculate_metric(
            y_prediction,
            y_test,
            return_value=show_logs,
        )
        if results is not None:
            for name, value in results:
                print(f"Step {i}: {name} --- {value}")

    for key, value in validation_model.calculate_metrics_mean():
        print(f"Average {key} during training: {value}")


# Example of using
def main():
    df = read_csv("../data/preprocessed/sales_train_preprocessed.csv")
    val_model = ModelValidation(df, "item_cnt_month")

    # use 33 because we have data of 33 month
    splitter = val_model.split_data(n_splits=33, window_type="sliding", train_size=3)
    model = LinearRegression()
    fit_validate(model, splitter, [RMSE, MAE, MSE], val_model)


if __name__ == "__main__":
    main()
