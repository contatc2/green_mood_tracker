from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud, ImageColorGenerator
from green_mood_tracker.data import clean
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from PIL import Image
import requests


def vectorizer(df, column):
    corpus = df[column].tolist()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    tf_idf = pd.DataFrame(
        data=X.toarray(), columns=vectorizer.get_feature_names())

    return tf_idf, X


# FINDING OPTIMAL K

def randomized_search(df, column):
    tf_idf, X = vectorizer(df, column)
    parameters = {
        'n_clusters': range(1, 20),
        'n_init': [10, 20, 30],
        'max_iter': [300, 500, 1000]
    }
    rs_search = RandomizedSearchCV(KMeans(), parameters, n_jobs=-1,
                                   verbose=1,
                                   refit=True, cv=5)
    rs_search.fit(X)
    return tf_idf, rs_search.best_params_


def grid_search(df, column):
    tf_idf, X = vectorizer(df, column)
    parameters = {
        'n_clusters': range(1, 20),
        'n_init': [10, 20, 30],
        'max_iter': [300, 500, 1000]
    }
    grid_search = GridSearchCV(KMeans(), parameters, n_jobs=-1,
                               verbose=1,
                               refit=True, cv=5)
    grid_search.fit(X)
    return tf_idf, grid_search.best_params_

# MOST COMMON WORDS BY CLUSTER


def run_KMeans(df, column):

    tf_idf, rs_optimal = randomized_search(df, column)
    tf_idf, grid_optimal = grid_search(df, column)

    min_k_list = []
    min_k_list.append(rs_optimal['n_clusters'])
    min_k_list.append(grid_optimal['n_clusters'])
    max_k = (min(min_k_list))

    max_init_list = []
    max_init_list.append(rs_optimal['n_init'])
    max_init_list.append(grid_optimal['n_init'])
    n_init = (max(max_init_list))

    max_k += 1
    kmeans_results = dict()
    for k in range(2, max_k):
        kmeans = cluster.KMeans(n_clusters=k, init='k-means++', n_init=n_init,
                                tol=0.0001, n_jobs=-1, random_state=42, algorithm='full',)
        kmeans_results.update({k: kmeans.fit(tf_idf)})

    return tf_idf, kmeans_results


def get_top_features_cluster(df, column, n_feats):
    tf_idf, kmeans_results = run_KMeans(df, column)
    tf_idf_array = tf_idf.to_numpy()
    kmeans = kmeans_results.get(list(kmeans_results.keys())[-1])
    prediction = kmeans.predict(tf_idf)

    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction == label)  # indices for each cluster
        # returns average score across cluster
        x_means = np.mean(tf_idf_array[id_temp], axis=0)
        # indices with top 20 scores
        sorted_means = np.argsort(x_means)[::-1][:n_feats]
        features = list(tf_idf.columns)
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns=['features', 'score'])
        dfs.append(df)
    return dfs, kmeans


def plotWords(df, column, n_feats):
    dfs, kmeans = get_top_features_cluster(df, column, n_feats)
    plt.figure(figsize=(8, 4))
    for i in range(0, len(dfs)):
        plt.title(("Most Common Words in Cluster {}".format(i)),
                  fontsize=10, fontweight='bold')
        sns.barplot(x='score', y='features', orient='h', data=dfs[i][:n_feats])
        plt.show()

# WORDCLOUDS


def Centroids(df, column, n_feats):
    tf_idf, kmeans_results = run_KMeans(df, column)
    dfs, kmeans = get_top_features_cluster(df, column, n_feats)
    centroids = pd.DataFrame(kmeans.cluster_centers_)
    centroids.columns = tf_idf.columns
    return centroids


def centroidsDict(df, column, index, n_feats, centroids):
    a = centroids.T[index].sort_values(ascending=False).reset_index().values
    centroid_dict = dict()

    for i in range(0, len(a)):
        centroid_dict.update({a[i, 0]: a[i, 1]})

    return centroid_dict


def generateWordClouds(df, column, n_feats, url=False):
    centroids = Centroids(df, column, n_feats)
    wordcloud = WordCloud(
        max_font_size=100, background_color='white', mask=mask)
    if url == False:
        for i in range(0, len(centroids)):
            centroid_dict = centroidsDict(df, column, i, n_feats, centroids)
            wordcloud.generate_from_frequencies(centroid_dict)
            plt.figure()
            plt.title('Cluster {}'.format(i))
            plt.imshow(wordcloud)
            plt.axis("off")
    else:
        mask = np.array(Image.open(requests.get(url, stream=True).raw))
        for i in range(0, len(centroids)):
            centroid_dict = centroidsDict(df, column, i, n_feats, centroids)
            wordcloud.generate_from_frequencies(centroid_dict)
            plt.figure()
            plt.title('Cluster {}'.format(i))
            plt.imshow(wordcloud)
            plt.axis("off")
    plt.show()


def get_lda(df, column, n_components, max_iter):
    tf_idf, X = vectorizer(df, column)
    params = {'n_components': n_components,
              'learning_decay': [.5, 0.7, 0.9],
              'max_iter': max_iter
              }
    lda = LatentDirichletAllocation()
    lda_search = GridSearchCV(lda, param_grid=params)
    lda_search.fit(X)
    # Best Model
    best_lda_model = lda_search.best_estimator_
    # Model Parameters
    params = lda_search.best_params_
    # Log Likelihood Score
    score = lda_search.best_score_
    # Perplexity
    perplexity = best_lda_model.perplexity(tf_idf)

    df_topic_keywords = pd.DataFrame(best_lda_model.components_)
    df_topic_keywords.columns = tf_idf.columns
    topicnames = ["Topic " + str(i)
                  for i in range(df_topic_keywords.index[-1]+1)]
    df_topic_keywords.index = topicnames

    return df_topic_keywords


def lda_wordcloud(df, column, n_components, max_iter, url=False):
    df_topics = get_lda(df, column, n_components, max_iter)
    dict_topics = df_topics.to_dict('records')
    if url == False:
        for i in range(len(dict_topics)):
            wordcloud = WordCloud(
                background_color='white').generate_from_frequencies(dict_topics[i])
            plt.figure(figsize=(10, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
    else:
        mask = np.array(Image.open(requests.get(url, stream=True).raw))
        for i in range(len(dict_topics)):
            wordcloud = WordCloud(
                background_color='white', mask=mask).generate_from_frequencies(dict_topics[i])
            plt.figure(figsize=(10, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")

    plt.show()


if __name__ == '__main__':
    main()
