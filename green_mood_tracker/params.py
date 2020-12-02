# GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'green-mood-tracker-01'

# Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
TWINT_TEST_FILE = 'green_energy_test.csv'

# Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

# Model - - - - - - - - - - - - - - - - - - - - - - - -

MODELS_FOLDER = 'models'
# model folder name (will contain the folders for all trained model versions)
ROBERTA_MODEL = 'RoBERTa'
MODEL_NAME = 'RoBERTa'
ROBERTA_FILENAME = 'roBERTa.tf'
WORD2VEC_FILENAME = 'word2vec.h5'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v0'


# model params
MAX_LENGTH = 50
