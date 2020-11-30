import os
import pandas as pd

from green_mood_tracker.gcp import download_model
from green_mood_tracker.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION, TWINT_TEST_FILE
from sklearn.metrics import accuracy_score


def get_prediction_data(filename):
    """
    Download twint data saved on GCP for prediction
    """
    path = "gs://{}/{}/{}/{}".format(BUCKET_NAME, 'data', 'twint_data', filename)
    df = pd.read_csv(path)
    return df

def get_test_data():
    """
    Download gold standard set
    """
    path = "gs://{}/{}/{}/{}".format(BUCKET_NAME, 'data', 'twint_data', TWINT_TEST_FILE)
    df = pd.read_csv(path)
    return df

def evaluate_model(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    return accuracy


def generate_prediction(model_name=MODEL_NAME, model_version=MODEL_VERSION):
    data = get_prediction_data()
    model = download_model(model_name=model_name, model_version=model_version)
    # encode prediction data
    if model_name == 'RoBERTa':
        encoder = RobertaEncoder()
        data = clean(data)
    else:
        encoder = Word2VecEncoder()
    # predict
    X = data.text
    y_pred = model.predict(encoder.transform(X))
    data["polarity"] = y_pred
    # df_sample = df_test[["key", "fare_amount"]]
    # name = f"predictions_{}{model_version}.csv"
    # data.to_csv(name, index=False)
    print("prediction saved")


def evaluate_model_on_gold_standard(model_name=MODEL_NAME, model_version=MODEL_VERSION):
    y = get_test_data().polarity
    y_pred = generate_prediction(model_name, model_version)
    return evaluate_model(y, y_pred)

if __name__ == '__main__':
    model_version = MODEL_VERSION
    evaluate_model_on_gold_standard(model_name=MODEL_NAME, model_version=MODEL_VERSION)
