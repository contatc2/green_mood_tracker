import os

from termcolor import colored
from tensorflow.keras.models import load_model
from transformers import TFRobertaForSequenceClassification

from green_mood_tracker.params import BUCKET_NAME, MODELS_FOLDER, MODEL_NAME, MODEL_VERSION, ROBERTA_FILENAME, ROBERTA_MODEL, WORD2VEC_FILENAME


def storage_upload_models(bucket_name=BUCKET_NAME, model_name=MODEL_NAME, model_version=MODEL_VERSION, model_filename=ROBERTA_FILENAME, rm=False):

    print(f'Uploading {model_filename}!')

    saved_model_path = os.path.join(MODELS_FOLDER, model_filename)

    if model_name == ROBERTA_MODEL:
        storage_location = '{}/{}/{}'.format(
            MODELS_FOLDER,
            model_name,
            model_version
        )
        command = f'gsutil -m cp -R {saved_model_path} gs://{bucket_name}/{storage_location}'
    else:
        storage_location = '{}/{}/{}/{}'.format(
            MODELS_FOLDER,
            model_name,
            model_version,
            model_filename
        )
        command = f'gsutil -m cp {saved_model_path} gs://{bucket_name}/{storage_location}'
    os.system(command)

    print(colored("=> {} uploaded to bucket {} inside {}".format(model_filename, bucket_name, storage_location),
                  "green"))
    if rm:
        os.system(f'rm -r {saved_model_path}')


def storage_upload_data(filename, folder='twint_data', bucket_name=BUCKET_NAME, rm=False):
    data_path = os.path.join(os.path.abspath(
        os.path.dirname(__file__)), 'raw_data', filename)

    storage_location = '{}/{}/{}'.format(
        'data',
        folder,
        filename
    )

    command = f'gsutil -m cp {data_path} gs://{bucket_name}/{storage_location}'
    os.system(command)

    print(colored("=> {} uploaded to bucket {} inside {}".format(
        filename, BUCKET_NAME, storage_location), "green"))

    if rm:
        os.remove(data_path)


def download_model_files(bucket_name=BUCKET_NAME, model_name=MODEL_NAME, model_version=MODEL_VERSION, model_filename='roBERTa.tf'):

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


def load_model(model_name=MODEL_NAME, rm=False):
    root = MODELS_FOLDER

    if model_name == ROBERTA_MODEL:
        saved_model_path = os.path.join(root, ROBERTA_FILENAME)
        model = TFRobertaForSequenceClassification.from_pretrained(
            saved_model_path)
    else:
        saved_model_path = os.path.join(root, WORD2VEC_FILENAME)
        model = load_model(saved_model_path)

    print(colored(f"=> loaded model {model_filename}", 'green'))

    if rm:
        os.system(f'rm -r {saved_model_path}')
    return model


if __name__ == '__main__':
    # download_model_files()
    storage_upload_models(bucket_name=BUCKET_NAME, model_name='word2vec', model_version=MODEL_VERSION, model_filename='roBERTa.tf', rm=False)
