# Project Overview

## Objective
 Understanding public sentiment toward energy transition through classification of tweets’ polarity in US and UK using NLP deep learning models
 The app has two functions:
 - visualise the sentiment on specific energy topics (-list topics) per region (UK/USA) and per year: map visualisation, sentiment evolution monthly, sentiment per tweet and sentiment per tweet weighted with likes (conveying general agreement with tweets)
 - perform a live analysis of sentiment for a topic, region and timeframe, scraping twitter data and using our NLP model to analyse sentiment.

## Team
- Henry Hall
- Clementine Contat
- Leonardo Gavaudan
- Thomas Gianetti

## Methods
- Data scraping
- Deep Learning
- Unsupervised Learning
- Data Visualisation

## Technology/Package used
- Twint API
- NLTK
- hugginface transformers RoBERTa
- Gensim word2vec
- Tensorflow Keras RNN
- Scikit-Learn
- Streamlit
- deployed on Google Cloud Platform
- K-means
- Latent Dirichlet Allocation
- Altoid
- Wordcloud

# Project Description

## Methodology
1 Identify datasets to train NLP models to identify polarity of tweets ( positive sentiment or negative sentiment)
2 Data cleaning using NLTK library
3 Train NLP models on sentiments datasets above (see models section)
4 Extract tweets by city, topics, start date, end date using the twint API
5 Create a test dataset of 200 tweets manually labelled to assess the model accuracy on energy tweets
6 Use trained model to automatically label extracted tweets with positive or negative polarity
7 Data visualisation using labelled tweets by location, topic and date
8 Deploy streamlit app using GCP virtual machine

## Training datasets
- sentiment140 - link - this dataset was excluded as the labelling methodology was only using emojis (vs manual labelling for other datasets) and was not creating satisfactory accuracy with our models
- STSGold - link
- kaggleSentiment - link
- other dataset - link

## Models
- huggingface/transformers/RoBERTa, using distill roberta-base to speed up live analysis
test-accuracy (using train_test_split on training datasets), accuracy on twint gold standard
- Keras RNN using a word2vec pretarined embedding using glove-twitter-100 library
test-accuracy (using train_test_split on training datasets), accuracy on twint gold standard

Other models tested:
- RNN with first layer embedding
- CNN with first layer embedding
- RNN with word2vec embedding trained on training data

## Architecture of the project

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
  $ sudo apt-get install virtualenv python-pip python-dev
  $ deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Check for green_mood_tracker in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/green_mood_tracker`
- Then populate it:

```bash
  $ ##   e.g. if group is "{group}" and project_name is "green_mood_tracker"
  $ git remote add origin git@gitlab.com:{group}/green_mood_tracker.git
  $ git push -u origin master
  $ git push -u origin --tags
```

Functionnal test with a script:
```bash
  $ cd /tmp
  $ green_mood_tracker-run
```
# Install
Go to `gitlab.com/{group}/green_mood_tracker` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:
```bash
  $ sudo apt-get install virtualenv python-pip python-dev
  $ deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:
```bash
  $ git clone gitlab.com/{group}/green_mood_tracker
  $ cd green_mood_tracker
  $ pip install -r requirements.txt
  $ make clean install test                # install and test
```
Functionnal test with a script:
```bash
  $ cd /tmp
  $ green_mood_tracker-run
``` 

# Continus integration
## Github 
Every push of `master` branch will execute `.github/workflows/pythonpackages.yml` docker jobs.
## Gitlab
Every push of `master` branch will execute `.gitlab-ci.yml` docker jobs.
