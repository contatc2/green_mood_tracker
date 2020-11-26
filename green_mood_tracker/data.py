import pandas as pd
from google.cloud import storage
from green_mood_tracker.training_data import get_raw_data
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
download('wordnet')
download('stopwords')
download('punkt')


def get_data(nrows=10000, local=False, binary=True, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    if local:
        return get_raw_data('raw_data/', binary)
    else:
        # binary or not
        # which data source do we want to keep?
        path = "gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH)
        df = pd.read_csv(path, nrows=nrows)
        return df

def clean(df, column):

    cachedStopWords = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    df[column] = df[column].apply(lambda x1: " ".join(
        filter(lambda x2: x2[0] != '@', x1.split())))
    df[column] = df[column].apply(lambda x: x.translate(
        str.maketrans('', '', string.punctuation)))
    df[column] = df[column].apply(lambda x: x.translate(
        str.maketrans('', '', string.digits)))
    df[column] = df[column].apply(lambda x: x.lower())
    df[column] = df[column].apply(lambda x: ' '.join(
        [lemmatizer.lemmatize(word) for word in x.split()]))
    df[column] = df[column].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in cachedStopWords]))

    return df


if __name__ == '__main__':
    params = dict(nrows=10000,
                  local=True,  # set to False to get data from GCP (Storage or BigQuery)
                  binary=True)
    df = get_data(**params)
    print(df.shape)
