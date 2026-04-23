from scripts.ETL import main as preprocess_data
from pandas import read_csv
from scripts.Validation import ModelValidation, fit_validate, RMSE, MAE, MSE
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    print("Start Preprocessing data".center(40, "="))
    preprocess_data("data")

    print("Start fit and validate model".center(40, "="))
    df = read_csv("data/preprocessed/sales_train_preprocessed.csv")
    val_model = ModelValidation(df, ["item_cnt_month"])

    # use 33 because we have data of 33 month
    splitter = val_model.split_data(n_splits=33, window_type="expanding")
    model = LinearRegression()
    fit_validate(model, splitter, [RMSE, MAE, MSE], val_model, True)
