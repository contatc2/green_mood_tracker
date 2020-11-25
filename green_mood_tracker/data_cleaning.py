import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
download('wordnet')


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


def main():
    pass


if __name__ == '__main__':
    main()
