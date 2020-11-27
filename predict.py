import os
import pandas as pd

from green_mood_tracker.gcp import download_model
from green_mood_tracker.params import MODEL_VERSION
from sklearn.metrics import recall, precision, accuracy


def get_prediction_data():
    """
    """
    # Add Client() here
    # path = "gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH)
    df = pd.read_csv(path)
    return df

def get_test_data():
    """
    """
    # Add Client() here
    # path = "gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH)
    df = pd.read_csv(path)
    return df


def evaluate_model(y, y_pred):
    # MAE = round(mean_absolute_error(y, y_pred), 2)
    # RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'Accuracy': accuracy, 'Precision': precision}
    return res


def generate_submission_csv(model_version=MODEL_VERSION):
    df_test = get_prediction_data()
    model = download_model(model_version)
    # Check if model savec was the ouptut of RandomSearch or Gridsearch
    y_pred = model.predict(df_test)
    # df_test["fare_amount"] = y_pred
    # df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_{model_version}.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved")


if __name__ == '__main__':
    model_version = MODEL_VERSION
    # model = download_model(folder)
    generate_submission_csv(model_version=model_version)
