# GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'green-mood-tracker-01'

# Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
TWINT_TEST_FILE = 'green_energy_test.csv'
DATA_FOLDER = 'data'
TWINT_FOLDER = 'twint_data'

# Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

# Model - - - - - - - - - - - - - - - - - - - - - - - -

MODELS_FOLDER = 'models'
# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'RoBERTa'
ROBERTA_MODEL = 'RoBERTa'
ROBERTA_FILENAME = 'roBERTa.tf'
WOR2VEC_MODEL = 'word2vec'
WORD2VEC_FILENAME = 'word2vec.h5'

# model version folder name (where the trained model.joblib file will be stored)

MODEL_VERSION = 'v0'

# model params
MAX_LENGTH = 50
