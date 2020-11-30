import time
import warnings
import os
from termcolor import colored

from green_mood_tracker.data import get_data
from green_mood_tracker.mlflow_trainer import MlFlowTrainer
from green_mood_tracker.encoders import Word2VecEncoder
from green_mood_tracker.params import MODEL_VERSION
from green_mood_tracker.gcp import storage_upload_models
from green_mood_tracker.utils import simple_time_tracker

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers


BATCH_SIZE = 32
# learning_rate = 7e-5
# epsilon = 1e-8
NUM_OF_EPOCHS = 1
PATIENCE = 5


class Word2VecTrainer(MlFlowTrainer):

    def __init__(self, X, y, **kwargs):

        super().__init__(kwargs['experiment_name'], kwargs['mlflow'])
        self.model = None
        self.kwargs = kwargs
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)
        self.val_split = self.kwargs.get("val_split", False)
        if self.split:
            self.X_train, self.X_test, self.y_train, self.y_test =\
                train_test_split(self.X_train, self.y_train,
                                 test_size=0.3, random_state=0)
            if self.val_split:
                self.X_train, self.X_val, self.y_train, self.y_val =\
                    train_test_split(
                        self.X_train, self.y_train, test_size=0.3, random_state=0)

        self.history = None
        self.rm = self.kwargs.get("rm", True)

    def create_embedding(self):
        # How can we use a pipeline here?
        # encoded modified features with tokenizer and added batch size
        encoder = Word2VecEncoder()
        self.X_train = encoder.fit_transform(self.X_train)
        if self.split:
            self.X_test = encoder.transform(self.X_test)
        if self.val_split:
            self.X_val = encoder.transform(self.X_val)


    def build_estimator(self):
        model = models.Sequential()
        model.add(layers.Masking())
        model.add(layers.GRU(units=32, activation='tanh', return_sequences=True))
        model.add(layers.GRU(units=16, activation='tanh'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss= 'binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        self.model = model

    @simple_time_tracker
    def train(self):
        # how do we want to pass the number of epochs
        tic = time.time()
        self.build_estimator()
        self.create_embedding()
        early_stop = EarlyStopping(
            patience=PATIENCE, restore_best_weights=True, monitor='val_accuracy'
            )
        self.history = self.model.fit(self.X_train, self.y_train, epochs=NUM_OF_EPOCHS,
                                 batch_size=BATCH_SIZE, validation_split=2/7, callbacks=[early_stop])
        self.mlflow_log_metric("train_time", int(time.time() - tic))

    def evaluate(self):
        accuracy = self.model.evaluate(self.X_train, self.y_train)[1]
        self.mlflow_log_metric("train_accuracy", accuracy)
        if self.split:
            accuracy_test = self.model.evaluate(self.X_test, self.y_test)[1]
            self.mlflow_log_metric("test_accuracy", accuracy_test)
            print(colored("accuracy train: {} || accuracy test: {}".format(
                accuracy, accuracy_test), "blue"))
        else:
            print(colored("accuracy train: {}".format(accuracy), "blue"))

    def save_model(self, upload=True, **kwargs):
        """Save the model and upload it on Google Storage /models folder
        """
        root = 'models'
        if not os.path.isdir(root):
            os.mkdir(root)

        model_filename = 'word2vec.h5'
        self.model.save(os.path.join(root, model_filename))
        print(colored("wor2vec.h5 saved locally", "green"))

        if upload:
            storage_upload_models(model_name='word2vec', model_version=MODEL_VERSION,
                                  model_filename=model_filename, rm=self.rm)

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    EXPERIMENT = "[GB] [London] [green_mood_tracker] word2vec"

    params = dict(nrows=100,
                  upload=True,
                  local=False,  # set to False to get data from GCP
                  mlflow=True, # set to True to log params to mlflow
                  rm=False,
                  experiment_name=EXPERIMENT
                  )

    print("############   Loading Data   ############")
    df = get_data(**params)
    y_train = df.polarity
    X_train = df.text
    del df
    print("shape: {}".format(X_train.shape))
    print("size: {} Mb".format(X_train.memory_usage() / 1e6))
    # Train and save model, locally and
    t = Word2VecTrainer(X=X_train, y=y_train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model(**params)
