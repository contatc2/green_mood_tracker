
from transformers import RobertaTokenizerFast
import tensorflow_datasets as tfds
# from tensorflow.data.Dataset import from_tensor_slices
from green_mood_tracker.utils import map_example_to_dict
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
        roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
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
