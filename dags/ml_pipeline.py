from model_kaggle_vaveyko import fit_model, ETL, make_predict
from airflow.sdk import task, dag
import joblib
import os


DATA_DIR = "/opt/airflow/data"
PREPROCESSED_DIR = "/opt/airflow/data/preprocessed"


@dag
def ml_pipeline():
    @task
    def preprocess_data():
        print("Start preprocessing")
        ETL(DATA_DIR, PREPROCESSED_DIR, False)
        print("Preprocessing completed")
        return 1

    @task
    def train_model(_):
        print("Start fitting")
        model = fit_model(f"{PREPROCESSED_DIR}/sales_train_preprocessed.csv")
        print("fitting completed")

        model_path = os.path.join(DATA_DIR, "random_forest_model.joblib")
        joblib.dump(model, model_path)
        print(f"Save into file {model_path}")
        return model_path

    @task
    def predict(model_path_name):
        model = joblib.load(model_path_name)
        make_predict(
            model,
            f"{PREPROCESSED_DIR}/test_preprocessed.csv",
            f"{PREPROCESSED_DIR}/sales_test_answer.csv",
        )

    data = preprocess_data()
    model_path = train_model(data)
    predict(model_path)


ml_pipeline()
