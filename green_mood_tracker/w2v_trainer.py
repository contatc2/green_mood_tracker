import time
import warnings
from termcolor import colored

from green_mood_tracker.data import get_data, clean
from green_mood_tracker.mlflow_trainer import MlFlowTrainer
from green_mood_tracker.encoders import Word2VecEncoder
from green_mood_tracker.params import MODEL_VERSION
from green_mood_tracker.gcp import storage_upload_models
from green_mood_tracker.utils import simple_time_tracker

from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


BATCH_SIZE = 32
# learning_rate = 7e-5
# epsilon = 1e-8
NUM_OF_EPOCHS = 3
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

        self.gridsearch = self.kwargs.get('gridsearch', False)
        self.model_params = None
        self.history = None

    def create_embedding(self):
        # How can we use a pipeline here?
        # encoded modified features with tokenizer and added batch size
        self.X_train = Word2VecEncoder().transform(self.X_train)
        if self.split:
            self.X_test = Word2VecEncoder().transform(self.X_test)
        if self.val_split:
            self.X_val = Word2VecEncoder().transform(self.X_val)


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
        self.model_params = {}

    def add_grid_search(self):
        """"
        Apply Gridsearch on self.params defined in get_estimator

        """
        # Here add randomseearch to your pipeline
        self.model = scikit_learn(self.model)
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
    def train(self):
        # how do we want to pass the number of epochs
        tic = time.time()
        self.build_estimator()
        self.create_embedding()
        if self.gridsearch:
            self.add_grid_search()
        early_stop = EarlyStopping(
            patience=PATIENCE, restore_best_weights=True, monitor='val_accuracy'
            )
        self.history = self.model.fit(self.X_train, self.y_train, epochs=NUM_OF_EPOCHS,
                                 batch_size=BATCH_SIZE, validation_split=2/7, callbacks=[early_stop])
        self.mlflow_log_metric("train_time", int(time.time() - tic))

    def evaluate(self):
        accuracy = self.model.evaluate(self.X_train)[1]
        self.mlflow_log_metric("train_accuracy", accuracy)
        if self.split:
            accuracy_test = self.model.evaluate(self.X_test)[1]
            self.mlflow_log_metric("test_accuracy", accuracy_test)
            if self.gridsearch:
                self.log_estimator_params()
            print(colored("accuracy train: {} || accuracy test: {}".format(
                accuracy, accuracy_test), "blue"))
        else:
            print(colored("accuracy train: {}".format(accuracy), "blue"))

    def save_model(self, upload=True, auto_remove=True, **kwargs):
        """Save the model and upload it on Google Storage /models folder
        """
        root = 'models/'
        model_filename = 'word2vec.h5'
        self.model.save(root+model_filename)
        print(colored("wor2vec.h5 saved locally", "green"))

        if upload:
            storage_upload_models(model_name='word2vec', model_version=MODEL_VERSION,
                                  model_filename=model_filename, rm=auto_remove)

    def log_estimator_params(self):
        # reg = self.get_estimator()
        # self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        # params = reg.get_params()
        # for k, v in params.items():
        #     self.mlflow_log_param(k, v)
        pass


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    EXPERIMENT = "[GB] [London] [green_mood_tracker] word2vec"

    params = dict(nrows=100,
                  upload=True,
                  local=False,  # set to False to get data from GCP
                  mlflow=True, # set to True to log params to mlflow
                  gridsearch=False,
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
    t = Word2VecTrainer(X=X_train, y=y_train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model(**params)
