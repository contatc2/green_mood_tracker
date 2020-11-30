import os
import joblib

from google.cloud import storage
from termcolor import colored

from green_mood_tracker.params import BUCKET_NAME, MODELS_FOLDER, MODEL_NAME, MODEL_VERSION



def storage_upload_models(bucket_name=BUCKET_NAME, model_name=MODEL_NAME, model_version=MODEL_VERSION, model_filename='model.joblib', rm=False):

    if model_name == 'RoBERTa':
        saved_model_path = os.path.join('models', 'RoBERTa.tf')
    else:
        saved_model_path =  os.path.join('models', model_filename)
    client = storage.Client().bucket(bucket_name)

    storage_location = '{}/{}/{}/{}'.format(
        MODELS_FOLDER,
        model_name,
        model_version,
        model_filename
    )

    blob = client.blob(storage_location)
    blob.upload_from_filename(filename=saved_model_path)
    print(colored("=> {} uploaded to bucket {} inside {}".format(model_filename, BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        shutil.rmtree(saved_model_path)


def storage_upload_data(filename, folder='twint_data', bucket=BUCKET_NAME, rm=False):
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


def download_model(bucket_name=BUCKET_NAME, model_name=MODEL_NAME, model_version=MODEL_VERSION, rm=True):
    client = storage.Client().bucket(bucket_name)
    if model_name == 'RoBERTa':
        model_filename = 'models/RoBERTa.tf/saved_model.pb'
    else:
        model_filename = 'word2vec.h5'

    saved_model_path = 'models/' + model_filename

    storage_location = '{}/{}/{}/{}'.format(
        MODELS_FOLDER,
        model_name,
        model_version,
        model_filename
    )
    blob = client.blob(storage_location)
    blob.download_to_filename(saved_model_path)
    print(f"=> {model_filename} downloaded from storage")
    model = joblib.load(model_name)
    if rm:
        os.remove(saved_model_path)
    return model
