import nest_asyncio
import sys
from twint_class import TWINT
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
download('wordnet')


def clean(df, column):

    cachedStopWords = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    df = pd.read_csv('../green_mood_tracker/data/sts_gold_tweet.csv', sep=';')
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


def main():
    pass


if __name__ == '__main__':
    main()
