from scripts.ETL import main as preprocess_data
from scripts.Model import fit_model, make_predict


if __name__ == "__main__":
    print("Start Preprocessing data".center(40, "="))
    preprocess_data("data")

    print("Start fit model".center(40, "="))
    model = fit_model("data/preprocessed/sales_train_preprocessed.csv")

    print("Start prediction".center(40, "="))
    make_predict(
        model,
        "data/preprocessed/test_preprocessed.csv",
        "data/preprocessed/sales_test_answer.csv",
    )

    print("FINISHED".center(40, "="))
