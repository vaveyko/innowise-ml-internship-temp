from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def fit_model(path_to_data: str) -> RandomForestRegressor:
    data = read_csv(path_to_data)
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        max_features="log2",
        min_samples_split=40,
        verbose=5,
        oob_score=True,
        n_jobs=-1,
    )

    X_train = data.drop(columns=["item_cnt_month", "date_block_num"])
    y_train = data[["item_cnt_month"]]

    model.fit(X_train, y_train["item_cnt_month"].to_list())
    return model


def make_predict(
    model: RandomForestRegressor, path_to_test_data: str, path_to_save_answer: str
) -> None:
    data_test = read_csv(path_to_test_data)
    answer = data_test[["ID"]]
    answer["ID"] = answer["ID"].astype(int)
    answer["item_cnt_month"] = model.predict(data_test.drop(columns="ID"))

    # target postprocessing
    answer["item_cnt_month"] = np.floor(answer["item_cnt_month"])

    answer.to_csv(path_to_save_answer, index=False)


def main(data_dir: str = "../data/preprocessed"):
    model = fit_model(data_dir + "/sales_train_preprocessed.csv")
    make_predict(
        model, data_dir + "/test_preprocessed.csv", data_dir + "/sales_test_answer.csv"
    )


if __name__ == "__main__":
    main()
