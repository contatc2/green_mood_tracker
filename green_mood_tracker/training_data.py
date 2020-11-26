import pandas as pd


def get_raw_data(path = 'raw_data/', binary=False):

    sts_gold = pd.read_csv(path + 'STS_gold.csv',sep=';')
    kaggle_sentiment_train = pd.read_csv(path + 'kaggle_sentiment_train.csv')
    kaggle_sentiment_test = pd.read_csv( path + 'kaggle_sentiment_test.csv')
    column_names = ['polarity','id','date','query','user','text']
    sentiment140 = pd.read_csv(path + 'training.1600000.processed.noemoticon.csv',encoding='latin-1',header=None,names=column_names)

    sts_gold_final = sts_gold[['id','tweet','polarity']].rename(columns={'tweet':'text'})
    sts_gold_final['source'] = "sts_gold"

    kaggle_sentiment_train['polarity'] = kaggle_sentiment_train.sentiment.map({'positive':4,'neutral':1,'negative':0})
    kaggle_sentiment_train_final = kaggle_sentiment_train[['textID','text','polarity']].rename(columns={'textID':'id'})
    kaggle_sentiment_train_final['source'] = "kaggle_sentiment_train"

    kaggle_sentiment_test['polarity'] = kaggle_sentiment_test.sentiment.map({'positive':4,'neutral':1,'negative':0})
    kaggle_sentiment_test_final = kaggle_sentiment_test[['textID','text','polarity']].rename(columns={'textID':'id'})
    kaggle_sentiment_test_final['source'] = "kaggle_sentiment_test"

    sentiment140_final = sentiment140[['id','text','polarity']]
    sentiment140_final['source'] = 'sentiment140'

    twitter_corpus = pd.read_csv(path + 'twitter_corpus.csv',error_bad_lines=False)
    twitter_corpus_final = twitter_corpus[['TweetId','TweetText','Sentiment']].rename(columns={'TweetId':'id','TweetText':'text','Sentiment':'polarity'})
    twitter_corpus_final['source'] = "twitter_corpus"
    twitter_corpus_final['polarity'] = twitter_corpus_final.polarity.map({'positive':4,'neutral':1,'negative':0})

    complete_data = pd.concat([sts_gold_final,kaggle_sentiment_train_final,kaggle_sentiment_test_final,twitter_corpus_final])
    complete_data = complete_data.dropna()

    if binary:
        complete_data_binary = complete_data[complete_data.polarity != 1]
        complete_data_binary['polarity'] = complete_data_binary.polarity.map({4:1,0:0})
        return complete_data_binary

    complete_data['polarity'] = complete_data.polarity.map({4:2,0:0, 1:1})
    return complete_data

def get_raw_data_notebook(binary=False):

    path = '../raw_data/'
    return get_raw_data(path, binary)


def get_raw_data_colab(binary=False):

    path = '/content/drive/My Drive/data_green_mood_tracker/'
    return get_raw_data(path, binary)


if __name__ == '__main__':
    raw_data = get_raw_data(binary=True)
    print(raw_data.shape)
