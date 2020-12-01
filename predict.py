import os
import pandas as pd
import numpy as np
from termcolor import colored

from green_mood_tracker.gcp import download_model_files, load_model
from green_mood_tracker.data import clean
from green_mood_tracker.encoders import RobertaEncoder, Word2VecEncoder
from green_mood_tracker.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION, TWINT_TEST_FILE
from sklearn.metrics import accuracy_score
import tensorflow as tf

BATCH_SIZE = 32


def get_prediction_data(data_filename):
    """
    Download twint data saved on GCP for prediction
    """
    path = "gs://{}/{}/{}/{}".format(BUCKET_NAME,
                                     'data', 'twint_data', data_filename)
    return pd.read_csv(path)


def get_test_data():
    """
    Download gold standard set
    """
    path = "gs://{}/{}/{}/{}".format(BUCKET_NAME,
                                     'data', 'twint_data', TWINT_TEST_FILE)
    return pd.read_csv(path)


def evaluate_model(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    print(colored(f'accuracy is {accuracy}', 'green'))
    return accuracy


def generate_prediction(data, model_name=MODEL_NAME, model_version=MODEL_VERSION, download_files=True):
    if download_files:
        download_model_files(model_name=model_name,
                             model_version=model_version)
    model = load_model(model_name=model_name)
    # encode prediction data
    if model_name == 'RoBERTa':
        encoder = RobertaEncoder(BATCH_SIZE)
        X = clean(data).text
        y = data.index
        ds_test_encoded = encoder.transform(X, y)
        results = tf.nn.softmax(model.predict(ds_test_encoded))
        y_pred = np.array(results).reshape((len(data), 2))[:, 1]
    else:
        encoder = Word2VecEncoder()
        X = data.text
        y_pred = model.predict(encoder.transform(X))
    return y_pred


def twint_prediction(data_filename, model_name=MODEL_NAME, model_version=MODEL_VERSION, download_files=False):
    data = get_prediction_data(data_filename)
    y_pred = generate_prediction(
        data, model_name=model_name, model_version=model_version, download_files=download_files)
    data["polarity"] = y_pred
    data_sample = data[["tweet", "polarity"]]
    name = f"predictions_{data_filename}{model_version}.csv"
    path = os.path.join('green_mood_tracker', 'raw_data', name)
    data_sample.to_csv(path, index=False)
    print("prediction saved")


def evaluate_model_on_gold_standard(model_name=MODEL_NAME, model_version=MODEL_VERSION, download_files=False):
    test_df = get_test_data()
    y = test_df.polarity
    y_pred = generate_prediction(test_df, model_name=model_name,
                                 model_version=model_version, download_files=download_files)
    y_pred = pd.Series(y_pred).map(lambda x: 1 if x >= 0.5 else 0)
    return evaluate_model(y, y_pred)


if __name__ == '__main__':
    evaluate_model_on_gold_standard(
        model_name=MODEL_NAME, model_version=MODEL_VERSION)
