import time
import joblib
import warnings
from termcolor import colored

from green_mood_tracker.data import get_data, clean
from green_mood_tracker.mlflow_trainer import MlFlowTrainer
from green_mood_tracker.encoders import RobertaEncoder
from green_mood_tracker.params import MODEL_VERSION
from green_mood_tracker.gcp import storage_upload
from green_mood_tracker.utils import simple_time_tracker

from sklearn.model_selection import train_test_split
from transformers import TFRobertaForSequenceClassification, AdamWeightDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


batch_size = 256
max_length = 30
learning_rate = 7e-5
epsilon = 1e-8
number_of_epochs = 10
patience = 5


class RobertaTrainer(MlFlowTrainer):

    def __init__(self, X, y, **kwargs):

        super().__init__(kwargs['experiment_name'], kwargs['mlflow'])
        self.model = None
        self.kwargs = kwargs
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)
        self.val_split = self.kwargs.get("val_split", True)
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

    def sentence_encode_all(self, batch_size=256):
        # How can we use a pipeline here?
        # encoded modified features with tokenizer and added batch size

        self.ds_train_encoded = RobertaEncoder(
            self.sentence_train, self.y_train).sentence_encode(batch_size, shuffle=True)
        if self.split:
            ds_test_encoded = RobertaEncoder(
                self.sentence_test, self.y_test).sentence_encode(batch_size)
        if self.val_split:
            ds_val_encoded = RobertaEncoder(
                self.sentence_val, self.y_val).sentence_encode(batch_size)

    def build_estimator(self, learning_rate=learning_rate, epsilon=1e-08):
        model = TFRobertaForSequenceClassification.from_pretrained(
            "roberta-base")
        optimizer = AdamWeightDecay(
            learning_rate=learning_rate, epsilon=epsilon, weight_decay=0)
        # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
        loss = SparseCategoricalCrossentropy(from_logits=True)
        metric = SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        self.model = model

    def add_grid_search(self):
        """"
        Apply Gridsearch on self.params defined in get_estimator

        """
        # Here add randomseearch to your pipeline
        grid_params = {'rgs__' + k: v for k, v in self.model_params.items()}

        self.model = RandomizedSearchCV(
            self.model,
            grid_params,
            n_iter=20,
            n_jobs=None,
            scoring=accuracy,
            cv=5,
        )

    @simple_time_tracker
    def train(self, number_of_epochs=number_of_epochs):
        # how do we want to pass the number of epochs
        tic = time.time()
        self.build_estimator()
        self.sentence_encode_all(batch_size=256)
        if self.gridsearch:
            self.add_grid_search()
        early_stop = EarlyStopping(
            patience=patience, restore_best_weights=True, monitor='val_accuracy')
        history = self.model.fit(self.ds_train_encoded, epochs=number_of_epochs,
                                 validation_data=ds_val_encoded, callbacks=[early_stop])
        # can we use a validation split here instead?
        self.mlflow_log_metric("train_time", int(time.time() - tic))
        return history

    def evaluate(self):
        accuracy = self.model.evaluate(self.ds_train_encoded)[1]
        self.mlflow_log_metric("train_accuracy", accuracy)
        if self.split:
            accuracy_test = self.model.evaluate(self.ds_test_encoded)[1]
            self.mlflow_log_metric("test_accuracy", accuracy_test)
            if self.gridsearch:
                self.log_estimator_params()
            print(colored("accuracy train: {} || accuracy test: {}".format(
                accuracy, accuracy_val), "blue"))
        else:
            print(colored("accuracy train: {}".format(accuracy), "blue"))

    def save_model(self, upload=True, auto_remove=True, **kwargs):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""

        # how do we put stuff in the models folder??? right path?
        model_name = 'model.joblib'
        joblib.dump(self.model, model_name)
        print(colored("model.joblib saved locally", "green"))

        if upload:
            storage_upload(model_version=MODEL_VERSION, rm=auto_remove)

    def log_estimator_params(self):
        # reg = self.get_estimator()
        # self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        # params = reg.get_params()
        # for k, v in params.items():
        #     self.mlflow_log_param(k, v)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    EXPERIMENT = "[GB] [London] [green_mood_tracker] RoBERTa"

    params = dict(nrows=10000,
                  upload=True,
                  local=False,  # set to False to get data from AWS
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
    print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
    # Train and save model, locally and
    t = RobertaTrainer(X=X_train, y=y_train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model(**params)
