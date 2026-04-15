from typing import Literal, Any, Callable, Iterable

from pandas import DataFrame, read_csv
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator


class ModelValidation:
    def __init__(
        self,
        data: DataFrame,
        y: str | list[str],
        x: str | list[str] = None,
    ):
        """
        :param data:
        :param y: target column name
        :param x: features columns names or all the rest except y
        """
        self.y_data = data[y]
        self.x_data = data.drop(y, axis=1) if x is None else data[x]
        self.metrics = {}

    def fit(
        self,
        model: BaseEstimator,
        splitter: Iterable,
        metric_func: list[Callable[[Any, Any], float]],
    ):
        """
        Fit model on each split of data and validate on test

        :param model:
        :param splitter:
        :param metric_func:
        :return:
        """

        self.metrics = {}
        for metric in metric_func:
            self.metrics[metric.__name__] = []

        for X_train, X_test, y_train, y_test in splitter:
            model.fit(X_train, y_train)
            y_prediction = model.predict(X_test)
            print(y_prediction)
            input()
            self.calculate_metric(y_prediction, y_test, metric_func)

        for key, value in self.metrics.items():
            print(f"Mean {key}: {np.mean(value)}")

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
        :param train_size: size of train subset in folds, train_size < n_split
        :yield: X_train, X_test, y_train, y_test
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
                max_train_size = train_size * len(self.x_data) // (n_splits + 1)

                splitter = TimeSeriesSplit(
                    n_splits=n_splits,
                    max_train_size=max_train_size,
                )

        for i, (train_idx, test_idx) in enumerate(splitter.split(self.x_data)):
            yield (
                self.x_data.iloc[train_idx],
                self.x_data.iloc[test_idx],
                self.y_data.iloc[train_idx],
                self.y_data.iloc[test_idx],
            )

    def calculate_metric(
        self,
        y_pred: Any,
        y_true: Any,
        metric_functions: list[Callable[[Any, Any], float]],
    ):
        """
        Calculate given metrics for current predictions

        :param y_pred:
        :param y_true:
        :param metric_functions:
        :return:
        """
        for metric_func in metric_functions:
            metric_name = metric_func.__name__
            metric = metric_func(y_pred, y_true)
            print(f"{metric_name}: {metric}")
            self.metrics[metric_name].append(metric)


# Metrics functions
def MSE(y_pred: Any, y_true: Any) -> float:
    return np.mean((y_pred - y_true) ** 2)


def RMSE(y_pred: Any, y_true: Any) -> float:
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def MAE(y_pred: Any, y_true: Any) -> float:
    return np.mean(np.abs(y_pred - y_true))


if __name__ == "__main__":
    df = read_csv("../data/preprocessed/sales_train_preprocessed.csv")
    val_model = ModelValidation(df, ["item_cnt_month"])

    # use 33 because we have data of 33 month
    splitter = val_model.split_data(n_splits=33, window_type="expanding")
    model = LinearRegression()
    val_model.fit(model, splitter, [MSE])

    # use 11 to split data bu 3 month
    splitter = val_model.split_data(n_splits=11, window_type="sliding")
    model = KNeighborsRegressor()
    val_model.fit(model, splitter, [RMSE, MAE])
