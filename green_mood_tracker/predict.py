import os
import pandas as pd
import numpy as np
import time
from termcolor import colored

from green_mood_tracker.gcp import download_model_files, load_local_model
from green_mood_tracker.data import clean
from green_mood_tracker.encoders import RobertaEncoder, Word2VecEncoder
from green_mood_tracker.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION, TWINT_TEST_FILE, MAX_LENGTH, DATA_FOLDER, TWINT_FOLDER, ROBERTA_MODEL
from sklearn.metrics import accuracy_score, f1_score
from green_mood_tracker.utils import simple_time_tracker
import tensorflow as tf
from pathlib import Path


def get_twint_data(data_filename, local=True, folder='raw_data'):
    """
    Download twint data saved on GCP for prediction
    """
    if local:
        path = os.path.join('green_mood_tracker', folder, data_filename)
    else:
        path = "gs://{}/{}/{}/{}".format(BUCKET_NAME,
                                         DATA_FOLDER, TWINT_FOLDER, data_filename)
    return pd.read_csv(path)


def evaluate_model(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(colored(f'accuracy: {accuracy}, f1: {f1}', 'green'))
    return accuracy, f1


def remove_csv_extension(csv_file):
    return csv_file.replace(".csv", "")


@simple_time_tracker
def encode_data(data, data_filename, model_name=MODEL_NAME):
    tic = time.time()
    if model_name == ROBERTA_MODEL:
        encoder = RobertaEncoder()
        X = clean(data, 'tweet').tweet
        y = data.index
        ds_encoded = encoder.transform(X, y)
    else:
        encoder = Word2VecEncoder()
        ds_encoded = encoder.transform(data.tweet)
    path = os.path.join('green_mood_tracker', 'raw_data',
                        f'{remove_csv_extension(data_filename)}_encoded')
    tf.data.experimental.save(ds_encoded, path)
    print(colored(f'encoding_time: {int(time.time() - tic)}', 'green'))
    return ds_encoded


@simple_time_tracker
def get_encoded_data(data_filename):
    tic = time.time()
    element_spec = ({'input_ids': tf.TensorSpec(shape=(None, MAX_LENGTH), dtype=tf.int32, name=None),
                     'attention_mask': tf.TensorSpec(shape=(None, MAX_LENGTH), dtype=tf.int32, name=None)},
                    tf.TensorSpec(shape=(None, 1), dtype=tf.int32, name=None))
    path = os.path.join('green_mood_tracker', 'raw_data')
    ds_filename = f'{remove_csv_extension(data_filename)}_encoded'
    ds_encoded = tf.data.experimental.load(
        os.path.join(path, ds_filename), element_spec)
    print(
        colored(f'retrieve encoding_time: {int(time.time() - tic)}', 'green'))
    return ds_encoded


@simple_time_tracker
def generate_prediction(data_filename, model, model_name=MODEL_NAME, binary=True):
    ds_encoded = get_encoded_data(data_filename)
    tic = time.time()
    if model_name == ROBERTA_MODEL:
        results = np.array(tf.nn.softmax(model.predict(ds_encoded).logits))
        y_pred = np.squeeze(results)[:, 1]
    else:
        y_pred = model.predict(ds_encoded)
    print(
        colored(f'generate prediction time: {int(time.time() - tic)}', 'green'))
    if binary:
        return pd.Series(y_pred).map(lambda x: 1 if x >= 0.5 else 0)
    return pd.Series(y_pred).map((lambda x: 2 if x >= 0.55 else (0 if x <= 0.45 else 1)))


@simple_time_tracker
def twint_prediction(data_filename, model_name=MODEL_NAME, model_version=MODEL_VERSION, download_gcp=False, encode=False, local=True):
    tic = time.time()
    tic_download = time.time()
    if download_gcp:
        download_model_files(model_name=model_name,
                             model_version=model_version)
    print(colored(
        f'download files from gcp time: {int(time.time() - tic_download)}', 'green'))

    tic_model = time.time()
    model = load_local_model(model_name=model_name)
    print(colored(f'load model time: {int(time.time() - tic_model)}', 'green'))
    data = get_twint_data(data_filename, local=local)
    if encode:
        encode_data(data, data_filename, model_name=model_name)
    y_pred = generate_prediction(
        data_filename, model, model_name=model_name)
    data["polarity"] = y_pred
    data_sample = data[["tweet", "polarity"]]
    name = f"predictions_{remove_csv_extension(data_filename)}{model_version}.csv"
    path = os.path.join('green_mood_tracker', 'raw_data', name)
    parent_dir = str(Path(path).parent)

    if(not os.path.isdir(parent_dir)):
        os.makedirs(parent_dir)
    data_sample.to_csv(path, index=False)
    print("prediction saved")
    print(colored(f'total prediction time: {int(time.time() - tic)}', 'green'))
    return data_sample


def evaluate_model_on_gold_standard(model_name=MODEL_NAME, model_version=MODEL_VERSION, download_gcp=False, encode=True, binary=True):
    data_filename = TWINT_TEST_FILE
    test_df = get_twint_data(data_filename)
    if encode:
        encode_data(test_df, data_filename, model_name=model_name)
    if binary:
        y_true = test_df.polarity
    else:
        y_true = test_df['polarity-neutral']
    if download_gcp:
        download_model_files(model_name=model_name,
                             model_version=model_version)
    model = load_local_model(model_name=model_name)
    y_pred = generate_prediction(
        data_filename, model, model_name=MODEL_NAME, binary=True)
    return evaluate_model(y_true, y_pred)


if __name__ == '__main__':
    # evaluate_model_on_gold_standard(
    #     model_name=MODEL_NAME, model_version='v2',
    #     download_gcp=False, encode=True, binary=True)
    # data_filename = 'city.csv'
    # data = get_twint_data(data_filename, local=True)
    # encode_data(data, data_filename, model_name=MODEL_NAME)
    # get_encoded_data(data_filename)
    # df = twint_prediction('city.csv', model_name=MODEL_NAME,
    #                       model_version=MODEL_VERSION, download_gcp=False,
    #                       encode=False)
    # print(df.head(5))
    twint_prediction('twint_test/uktest.csv', encode=True)
