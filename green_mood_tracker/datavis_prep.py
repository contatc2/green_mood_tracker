from green_mood_tracker.encoders import RobertaEncoder
from transformers import TFRobertaForSequenceClassification
import tensorflow as tf
import numpy as np
import pandas as pd
from vega_datasets import data
from green_mood_tracker.data import clean


model_load = TFRobertaForSequenceClassification.from_pretrained(
    'models/roBERTa.tf')

def cleantopic(df, topic="['solar', 'energy']"):

	df_topic = df[df['search'] == topic]

	df_clean = clean(df_topic, 'tweet')

	ds_twint_encoded = RobertaEncoder(batch_size=64).transform(
		df_clean.tweet, df_clean.timezone, shuffle=True)

	return ds_twint_encoded, df_clean


def results(ds_twint_encoded, df_clean):
	submission_pre = tf.nn.softmax(model_load.predict(ds_twint_encoded).logits)
	submission_pre = np.reshape(submission_pre.numpy(), (len(df_clean), 2))
	return submission_pre


def comment_dataframe_prep(df_clean, submission_pre):
	comment_dataframe = df_clean.copy()
	comment_dataframe['nlikes'] = comment_dataframe['nlikes'].copy() + 1
	comment_dataframe['prob_neg'] = 0
	comment_dataframe['prob_pos'] = 0
	comment_dataframe['prob_neg'] = submission_pre[:, 0]
	comment_dataframe['prob_pos'] = submission_pre[:, 1]
	comment_dataframe['label'] = comment_dataframe['prob_pos'].apply(
		(lambda x: 2 if x >= 0.55 else (0 if x <= 0.45 else 1)))
	comment_dataframe = comment_dataframe.drop_duplicates(
		subset=['id'], keep='first')
	comment_dataframe = comment_dataframe.drop_duplicates(
		subset=['date'], keep='first')
	comment_dataframe = comment_dataframe[comment_dataframe['date'] != '14502749']
	comment_dataframe['date'] = comment_dataframe[[
		'date']].apply(pd.to_datetime)
	return comment_dataframe[['date', 'tweet', 'nlikes', 'label', 'prob_neg', 'prob_pos', 'state_code']]


def all_plotting(topic="['solar', 'energy']"):
	us_twint = pd.read_csv('green_mood_tracker/raw_data/twint_US.csv',
						   dtype={"date": "string", "tweet": "string"})
	uk_twint = pd.read_csv('green_mood_tracker/raw_data/twint_data_UK.csv',
						   dtype={"date": "string", "tweet": "string"})

	ds_twint_encoded, df_clean = cleantopic(us_twint)
	submission_pre = results(ds_twint_encoded, df_clean)

	comment_dataframe = comment_dataframe_prep(df_clean, submission_pre)
	comment_dataframe.to_csv("green_mood_tracker/raw_data/US/solar.csv")


if __name__ == "__main__":
	#df = read_data()
	all_plotting(topic="['solar', 'energy']")
