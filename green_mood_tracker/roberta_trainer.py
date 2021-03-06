import time
import warnings
import os
from termcolor import colored

from green_mood_tracker.data import get_data, clean
from green_mood_tracker.mlflow_trainer import MlFlowTrainer
from green_mood_tracker.encoders import RobertaEncoder
from green_mood_tracker.params import MODEL_VERSION, MODELS_FOLDER, ROBERTA_FILENAME
from green_mood_tracker.gcp import storage_upload_models
from green_mood_tracker.utils import simple_time_tracker

from sklearn.model_selection import train_test_split
from transformers import TFRobertaForSequenceClassification, AdamWeightDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

BATCH_SIZE = 32
NUM_OF_EPOCHS = 30
PATIENCE = 5

LEARNING_RATE = 1e-05
EPSILON = 1e-08
BETA = 0.9
DECAY = 0.001
ROBERTA_BASE = 'distilroberta-base'

class RobertaTrainer(MlFlowTrainer):

    def __init__(self, X, y, **kwargs):

        super().__init__(kwargs['experiment_name'], kwargs['mlflow'])
        self.model = None
        self.kwargs = kwargs
        self.local = self.kwargs.get("local", True)
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)
        self.val_split = self.kwargs.get("val_split", True)
        self.rm = self.kwargs.get("rm", True)
        if self.split:
            self.sentence_train, self.sentence_test, self.y_train, self.y_test =\
                train_test_split(self.X_train, self.y_train,
                                 test_size=0.3, random_state=0)
            if self.val_split:
                self.sentence_train, self.sentence_val, self.y_train, self.y_val =\
                    train_test_split(
                        self.sentence_train, self.y_train, test_size=0.3, random_state=0)

        self.ds_train_encoded = None
        self.ds_test_encoded = None
        self.ds_val_encoded = None

    def sentence_encode_all(self, batch_size=BATCH_SIZE):
        # How can we use a pipeline here?
        # encoded modified features with tokenizer and added batch size
        encoder = RobertaEncoder(batch_size)

        self.ds_train_encoded = encoder.transform(
            self.sentence_train, self.y_train, shuffle=True)
        if self.split:
            self.ds_test_encoded = encoder.transform(
                self.sentence_test, self.y_test)
        if self.val_split:
            self.ds_val_encoded = encoder.transform(
                self.sentence_val, self.y_val)

    def build_estimator(self):
        model = TFRobertaForSequenceClassification.from_pretrained(
            ROBERTA_BASE)
        optimizer = AdamWeightDecay(
            learning_rate=LEARNING_RATE, epsilon=EPSILON, weight_decay_rate=DECAY, beta_1=BETA)
        # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
        loss = SparseCategoricalCrossentropy(from_logits=True)
        metric = SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        self.model = model

    @simple_time_tracker
    def train(self):
        # how do we want to pass the number of epochs
        tic = time.time()
        self.build_estimator()
        self.sentence_encode_all()
        early_stop = EarlyStopping(
            patience=PATIENCE, restore_best_weights=True, monitor='val_accuracy')
        history = self.model.fit(self.ds_train_encoded, epochs=NUM_OF_EPOCHS,
                                 validation_data=self.ds_val_encoded, callbacks=[early_stop])
        # can we use a validation split here instead?
        self.mlflow_log_metric("train_time", int(time.time() - tic))
        return history

    def evaluate(self):
        accuracy = self.model.evaluate(self.ds_train_encoded)[1]
        self.mlflow_log_metric("train_accuracy", accuracy)
        if self.split:
            accuracy_test = self.model.evaluate(self.ds_test_encoded)[1]
            self.mlflow_log_metric("test_accuracy", accuracy_test)
            print(colored("accuracy train: {} || accuracy test: {}".format(
                accuracy, accuracy_test), "blue"))
        else:
            print(colored("accuracy train: {}".format(accuracy), "blue"))

    def save_model(self, upload=True, **kwargs):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""

        root = MODELS_FOLDER
        if not os.path.isdir(root):
            os.mkdir(root)

        model_filename = ROBERTA_FILENAME
        self.model.save_pretrained(os.path.join(root, model_filename))
        print(colored(f"{ROBERTA_FILENAME} saved locally", "green"))

        if upload:
            storage_upload_models(model_name='RoBERTa', model_version=MODEL_VERSION,
                                  model_filename=model_filename, rm=self.rm)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    EXPERIMENT = "[GB] [London] [green_mood_tracker] RoBERTa"


    params = dict(nrows=None,
                  upload=False,
                  local=False,
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name=EXPERIMENT
                  )

    print("############   Loading Data   ############")
    df = get_data(**params)
    df = clean(df, 'text')
    y_train = df.polarity
    X_train = df.text
    del df
    print("shape: {}".format(X_train.shape))
    print("size: {} Mb".format(X_train.memory_usage() / 1e6))
    # Train and save model, locally and
    t = RobertaTrainer(X=X_train, y=y_train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model(**params)
