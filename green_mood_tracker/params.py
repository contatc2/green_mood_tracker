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

# Twint cities

UK_LIST = ['London',
           'Birmingham',
           'Leeds',
           'Glasgow',
           'Sheffield',
           'Bradford',
           'Edinburgh',
           'Liverpool',
           'Manchester',
           'Bristol',
           'Wakefield',
           'Cardiff',
           'Coventry',
           'Nottingham',
           'Leicester',
           'Sunderland',
           'Belfast',
           'Newcastle upon Tyne',
           'Brighton',
           'Hull',
           'Plymouth',
           'Stoke-on-Trent',
           'Wolverhampton',
           'Derby',
           'Swansea',
           'Southampton',
           'Salford',
           'Aberdeen',
           'Westminster',
           'Portsmouth',
           'York',
           'Peterborough',
           'Dundee',
           'Lancaster',
           'Oxford',
           'Newport',
           'Preston',
           'St Albans',
           'Norwich',
           'Chester',
           'Cambridge',
           'Salisbury',
           'Exeter',
           'Gloucester',
           'Lisburn',
           'Chichester',
           'Winchester',
           'Londonderry',
           'Carlisle',
           'Worcester',
           'Bath',
           'Durham',
           'Lincoln',
           'Hereford',
           'Armagh',
           'Inverness',
           'Stirling',
           'Canterbury',
           'Lichfield',
           'Newry',
           'Ripon',
           'Bangor',
           'Truro',
           'Ely',
           'Wells',
           'St Davids']


USA_LIST = ['New York',
            'Los Angeles',
            'Chicago',
            'Houston',
            'Philadelphia',
            'Phoenix',
            'San Antonio',
            'San Diego',
            'Dallas',
            'San Jose',
            'Austin',
            'Jacksonville',
            'San Francisco',
            'Indianapolis',
            'Columbus',
            'Fort Worth',
            'Charlotte',
            'Seattle',
            'Denver',
            'El Paso',
            'Detroit',
            'Washington',
            'Boston',
            'Memphis',
            'Nashville',
            'Portland',
            'Oklahoma City',
            'Las Vegas',
            'Baltimore',
            'Louisville',
            'Milwaukee',
            'Albuquerque',
            'Tucson',
            'Fresno',
            'Sacramento',
            'Kansas City',
            'Long Beach',
            'Mesa',
            'Atlanta',
            'Colorado Springs',
            'Virginia Beach',
            'Raleigh',
            'Omaha',
            'Miami',
            'Oakland',
            'Minneapolis',
            'Tulsa',
            'Wichita',
            'New Orleans',
            'Arlington']
