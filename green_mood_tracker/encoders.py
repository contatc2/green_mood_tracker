import numpy as np

from transformers import RobertaTokenizerFast
import tensorflow_datasets as tfds
# from tensorflow.data.Dataset import from_tensor_slices
from green_mood_tracker.utils import map_example_to_dict
from green_mood_tracker.data import clean_series
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin

import gensim.downloader as api
import tensorflow as tf

MAX_LENGTH = 30


class RobertaEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, batch_size):
        self.input_ids_list = []
        self.attention_mask_list = []
        self.label_list = []
        self.batch_size = batch_size
        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def convert_example_to_feature(self, entry):
        # combine step for tokenization, WordPiece vector mapping
        # add also special tokens and truncate reviews longer than our max length
        return self.roberta_tokenizer.encode_plus(entry,
                                             # add [CLS], [SEP]
                                             add_special_tokens=True,
                                             max_length=MAX_LENGTH,  # max length of text that can go to RoBERTa
                                             truncation=True,
                                             # add [PAD] tokens at the end of sentence
                                             padding='max_length',
                                             return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                             )

    # map to the expected input to TFRobertaForSequenceClassification, see here
    # def map_example_to_dict(self, input_ids, attention_masks, label):
    #     return {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_masks,
    #     }, label

    def encode_examples(self, ds, limit=-1):
        # Prepare Input list
        if limit > 0:
            ds = ds.take(limit)

        for entry, label in tfds.as_numpy(ds):
            bert_input = self.convert_example_to_feature(entry.decode())
            self.input_ids_list.append(bert_input['input_ids'])
            self.attention_mask_list.append(bert_input['attention_mask'])
            self.label_list.append([label])

        return tf.data.Dataset.from_tensor_slices((self.input_ids_list, self.attention_mask_list, self.label_list))\
            .map(map_example_to_dict)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, shuffle=False):
        # encoded modified features with tokenizer and added batch size

        #if y.any():
        
        sentences_modified = tf.data.Dataset.from_tensor_slices((X, y))

        #else:
        #sentences_modified = tf.data.Dataset.from_tensor_slices((X))

        if shuffle:
            return self.encode_examples(sentences_modified).shuffle(10000).batch(self.batch_size)
        return self.encode_examples(sentences_modified).batch(self.batch_size)


class Word2VecEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        self.library =kwargs.get('library', "glove-twitter-100")
        self.vectors = api.load(self.library)

    def embed_sentence(self, sentence):
        embedded_sentence = []
        for word in sentence:
            if word in self.vectors.vocab.keys():
                vector = self.vectors[word]
                embedded_sentence.append(vector)
        return np.array(embedded_sentence)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Sentences to list of words
        embedding = []
        for sentence in clean_series(X):
            embedded_sentence = self.embed_sentence(sentence)
            embedding.append(embedded_sentence)
        return pad_sequences(embedding, dtype='float32', padding='post')
