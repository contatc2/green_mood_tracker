import numpy as np

from transformers import RobertaTokenizer
import tensorflow_datasets as tfds
# from tensorflow.data.Dataset import from_tensor_slices
from green_mood_tracker.utils import map_example_to_dict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin

import gensim.downloader as api
import tensorflow as tf

MAX_LENGTH = 30


class RobertaEncoder():

    def __init__(self, sentence, y):
        self.sentence = sentence
        self.y = y
        self.input_ids_list = []
        self.attention_mask_list = []
        self.label_list = []

    def convert_example_to_feature(self, entry):
        # combine step for tokenization, WordPiece vector mapping
        # add also special tokens and truncate reviews longer than our max length
        roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        return roberta_tokenizer.encode_plus(entry,
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

    def sentence_encode(self, batch_size, shuffle=False):
        # encoded modified features with tokenizer and added batch size
        sentences_modified = tf.data.Dataset.from_tensor_slices(
            (self.sentence, self.y))

        if shuffle:
            return self.encode_examples(sentences_modified).shuffle(10000).batch(batch_size)
        return self.encode_examples(sentences_modified).batch(batch_size)


class Word2VecEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, sentences=None):
        self.sentences = sentences
        self.word2vec = None
        self.embedding = []

    def get_word2vec(self):
        self.word2vec = api.load("glove-twitter-100")

    def embed_sentence(self, sentence):
        embedded_sentence = []
        for word in sentence:
            if word in self.word2vec.wv.vocab.keys():
                vector = self.word2vec.wv[word]
                embedded_sentence.append(vector)
        return np.array(embedded_sentence)

    def embedding_pipeline(self):
        # Sentences to list of words
        self.get_word2vec()
        for sentence in self.sentences.map(lambda x: x.split()):
            embedded_sentence = self.embed_sentence(sentence)
            self.embedding.append(embedded_sentence)
        # Pad the inputs
        return pad_sequences(self.embedding, dtype='float32', padding='post')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Sentences to list of words
        self.get_word2vec()
        for sentence in X.map(lambda x: x.split()):
            embedded_sentence = self.embed_sentence(sentence)
            self.embedding.append(embedded_sentence)
        # Pad the inputs
        X = pad_sequences(self.embedding, dtype='float32', padding='post')
        return X
