import tensorflow as tf
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
import twint
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from green_mood_tracker.training_data import get_raw_data
from green_mood_tracker.data_cleaning import clean
import tensorflow_datasets as tfds
from transformers import TFRobertaForSequenceClassification
from transformers import RobertaTokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models

max_length = 30
batch_size = 256
max_length = 30
learning_rate = 7e-5
epsilon = 1e-8
number_of_epochs = 10
patience = 5
sample=True
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def data_prep(sample=True):
    raw_data = get_raw_data()
    raw_data_clean = clean(raw_data,'text')
    raw_data_clean = raw_data_clean[raw_data_clean['polarity']!=1]
    data_sample = raw_data_clean[raw_data_clean['source']!='sentiment140']
    data_sample['polarity'] = data_sample.polarity.map({2:1,0:0})

    if sample is True:
        data_sample = data_sample.sample(n=2_000,random_state=0).reset_index()

    X = data_sample.text
    y = data_sample.polarity
    sentence_train, sentence_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
    sentence_train, sentence_val, y_train, y_val = train_test_split(sentence_train, y_train, test_size=0.3, random_state = 0)
    return sentence_train, sentence_val, y_train, y_val, sentence_test, y_test


def convert_example_to_feature(entry):
    # combine step for tokenization, WordPiece vector mapping and will
    # add also special tokens and truncate reviews longer than our max length
    return roberta_tokenizer.encode_plus(entry,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to RoBERTa
                                 truncation=True,
                                 padding= 'max_length',  # add [PAD] tokens at the end of sentence
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 )

# map to the expected input to TFRobertaForSequenceClassification, see here
def map_example_to_dict(input_ids, attention_masks, label):
    return {
      "input_ids": input_ids,
      "attention_mask": attention_masks,
           }, label


def encode_examples(ds, limit=-1):
    # Prepare Input list
    input_ids_list = []
    attention_mask_list = []
    label_list = []

    if (limit > 0):
        ds = ds.take(limit)

    for entry, label in tfds.as_numpy(ds):
        bert_input = convert_example_to_feature(entry.decode())
        input_ids_list.append(bert_input['input_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices((input_ids_list,
                                               attention_mask_list,
                                               label_list)).map(map_example_to_dict)



# modify features into tensorflow slices
def sentence_encode():
    sentence_train, sentence_val, y_train, y_val, sentence_test, y_test = data_prep()


    training_sentences_modified = tf.data.Dataset.from_tensor_slices((sentence_train,
                                                                      y_train))
    val_sentences_modified = tf.data.Dataset.from_tensor_slices((sentence_val,
                                                                     y_val))
    testing_sentences_modified = tf.data.Dataset.from_tensor_slices((sentence_test,
                                                                     y_test))

    ## encoded modified features with tokenizer and added batch size

    ds_train_encoded = encode_examples(training_sentences_modified).shuffle(10000).batch(batch_size)
    ds_val_encoded = encode_examples(val_sentences_modified).batch(batch_size)
    ds_test_encoded = encode_examples(testing_sentences_modified).batch(batch_size)

    return ds_train_encoded, ds_val_encoded, ds_test_encoded


#build model and compile
def model_build(learning_rate=learning_rate,epsilon=1e-08):
    model = TFRobertaForSequenceClassification.from_pretrained("roberta-base")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
    # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model


# fit model
def fit_model_and_save(number_of_epochs = number_of_epochs,):
    model = model_build()
    ds_train_encoded, ds_val_encoded, ds_test_encoded = sentence_encode()
    es = EarlyStopping(patience=patience, restore_best_weights=True,monitor='val_accuracy')
    history = model.fit(ds_train_encoded, epochs=number_of_epochs,
              validation_data=ds_val_encoded, callbacks=[es])

    loss,accuracy =  model.evaluate(ds_test_encoded)
    print(accuracy)

    # You can save it :
    #models.save_model(model, 'enter_name_of_model')

if __name__ == "__main__":
    fit_model_and_save()



