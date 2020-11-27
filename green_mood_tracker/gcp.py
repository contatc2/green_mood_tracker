import os
import joblib
import sys

from google.cloud import storage
from termcolor import colored

from green_mood_tracker.params import BUCKET_NAME, BUCKET_FOLDER, MODEL_NAME, MODEL_VERSION



def storage_upload_models(bucket_name=BUCKET_NAME, model_name=MODEL_NAME, model_version=MODEL_VERSION, model_filename='model.joblib', rm=False):

    sys.path.insert(0, '../')
    saved_model_path = 'models/RoBERTa.tf/saved_model.pb'
    client = storage.Client().bucket(bucket_name)

    BUCKET_FOLDER = 'models'

    storage_location = '{}/{}/{}/{}'.format(
        BUCKET_FOLDER,
        model_name,
        model_version,
        model_filename
    )

    blob = client.blob(storage_location)
    blob.upload_from_filename(filename=saved_model_path)
    print(colored("=> {} uploaded to bucket {} inside {}".format(model_filename, BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove(saved_model_path)


def storage_upload_data(filename, folder='twint_data', bucket=BUCKET_NAME, rm=False):
    sys.path.insert(0, '../')
    root = 'raw_data/'
    file_path = root + filename
    client = storage.Client().bucket(bucket)
    storage_location = '{}/{}/{}'.format(
        'data',
        folder,
        filename
    )
    blob = client.blob(storage_location)
    blob.upload_from_filename(filename=file_path)
    print(colored("=> {} uploaded to bucket {} inside {}".format(filename, BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove(file_path)


def download_model(model_version=MODEL_VERSION, bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)
    model_name = 'model.joblib'
    storage_location = '{}/{}/{}/{}'.format(
        BUCKET_FOLDER,
        MODEL_NAME,
        model_version,
        model_name
    )
    blob = client.blob(storage_location)
    blob.download_to_filename(model_name)
    print(f"=> pipeline downloaded from storage")
    model = joblib.load(model_name)
    if rm:
        os.remove(model_name)
    return model
