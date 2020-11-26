import os
import joblib

from google.cloud import storage
from termcolor import colored

from green_mood_tracker.params import BUCKET_NAME, BUCKET_FOLDER, MODEL_NAME, MODEL_VERSION


def storage_upload(model_version=MODEL_VERSION, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)
    model_name = 'model.joblib'
    storage_location = '{}/{}/{}/{}'.format(
        BUCKET_FOLDER,
        MODEL_NAME,
        model_version,
        model_name
        )
    blob = client.blob(storage_location)
    blob.upload_from_filename(filename=model_name)
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')

def storage_upload_data( filename, bucket=BUCKET_NAME, rm=False):
    root_path = '../raw_data/'
    file_path = root_path + filename
    client = storage.Client().bucket(bucket)
    storage_location = '{}/{}'.format(
        'data',
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
