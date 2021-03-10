import os

from termcolor import colored
from tensorflow.keras.models import load_model
from transformers import TFRobertaForSequenceClassification
from green_mood_tracker.params import BUCKET_NAME, MODELS_FOLDER, MODEL_NAME, MODEL_VERSION, ROBERTA_FILENAME, ROBERTA_MODEL, WORD2VEC_FILENAME, WOR2VEC_MODEL, TWINT_FOLDER, DATA_FOLDER


def storage_upload_models(bucket_name=BUCKET_NAME, model_name=MODEL_NAME, model_version=MODEL_VERSION, model_filename=ROBERTA_FILENAME, rm=False):

    print(f'Uploading {model_filename}!')

    saved_model_path = os.path.join(MODELS_FOLDER, model_filename)
    storage_location = '{}/{}/{}/{}'.format(
        MODELS_FOLDER,
        model_name,
        model_version,
        model_filename
    )

    if model_name == ROBERTA_MODEL:
        command = f'gsutil -m cp -R {saved_model_path} gs://{bucket_name}/{storage_location}'
    else:
        command = f'gsutil -m cp {saved_model_path} gs://{bucket_name}/{storage_location}'
    os.system(command)

    print(colored("=> {} uploaded to bucket {} inside {}".format(model_filename, bucket_name, storage_location),
                  "green"))
    if rm:
        os.system(f'rm -r {saved_model_path}')


def storage_upload_data(filename, folder=TWINT_FOLDER, bucket_name=BUCKET_NAME, rm=False):
    data_path = os.path.join(os.path.abspath(
        os.path.dirname(__file__)), 'raw_data', filename)

    storage_location = '{}/{}/{}'.format(
        DATA_FOLDER,
        folder,
        filename
    )

    command = f'gsutil -m cp {data_path} gs://{bucket_name}/{storage_location}'
    os.system(command)

    print(colored("=> {} uploaded to bucket {} inside {}".format(
        filename, BUCKET_NAME, storage_location), "green"))

    if rm:
        os.remove(data_path)


def storage_download_data(filename, folder=TWINT_FOLDER, bucket_name=BUCKET_NAME, import_folder=True):
    data_path = os.path.join(os.path.abspath(
        os.path.dirname(__file__)), 'raw_data')

    storage_location = '{}/{}/{}'.format(
        DATA_FOLDER,
        folder,
        filename
    )
    folder_string = ' -R' if import_folder else ''

    command = f'gsutil -m cp{folder_string} gs://{bucket_name}/{storage_location} {data_path}'
    os.system(command)

    print(colored("=> {} downloaded from bucket {} storage {}".format(
        filename, BUCKET_NAME, storage_location), "green"))


def download_model_files(bucket_name=BUCKET_NAME, model_name=MODEL_NAME, model_version=MODEL_VERSION, model_filename=ROBERTA_FILENAME):

    storage_location = '{}/{}/{}/{}'.format(
        MODELS_FOLDER,
        model_name,
        model_version,
        model_filename
    )

    root = MODELS_FOLDER

    if not os.path.isdir(root):
        os.mkdir(root)

    if model_name == ROBERTA_MODEL:
        command = f'gsutil -m cp -R gs://{bucket_name}/{storage_location} {root}'
    else:
        command = f'gsutil -m cp gs://{bucket_name}/{storage_location} {root}'
    os.system(command)

    print(colored(f"=> {model_filename} downloaded from storage", 'green'))


def load_local_model(model_name=MODEL_NAME, rm=False):
    root = MODELS_FOLDER

    if model_name == ROBERTA_MODEL:
        model_filename = ROBERTA_FILENAME
        saved_model_path = os.path.join(root, model_filename)
        model = TFRobertaForSequenceClassification.from_pretrained(
            saved_model_path)
    else:
        model_filename = WORD2VEC_FILENAME
        saved_model_path = os.path.join(root, model_filename)
        model = load_model(saved_model_path)

    print(colored(f"=> loaded model {model_filename}", 'green'))

    if rm:
        os.system(f'rm -r {saved_model_path}')
    return model


if __name__ == '__main__':
    storage_download_data('UK', folder=TWINT_FOLDER,
                          bucket_name=BUCKET_NAME, import_folder=True)
    storage_download_data('US', folder=TWINT_FOLDER,
                          bucket_name=BUCKET_NAME, import_folder=True)
    download_model_files(bucket_name=BUCKET_NAME, model_name=ROBERTA_MODEL,
                         model_version='v2', model_filename=ROBERTA_FILENAME)
    download_model_files(bucket_name=BUCKET_NAME, model_name=WOR2VEC_MODEL,
                         model_version='v1', model_filename=WORD2VEC_FILENAME)
