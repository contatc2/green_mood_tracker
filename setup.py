from setuptools import find_packages
from setuptools import setup

# with open('requirements.txt') as f:
#     content = f.readlines()
# requirements = [x.strip() for x in content if 'git+' not in x]


REQUIRED_PACKAGES = [
    'pip>=9',
    'setuptools>=26',
    'wheel>=0.29',
    'pandas',
    'numpy',
    'pytest',
    'coverage',
    'flake8',
    'black',
    'yapf',
    'python-gitlab',
    'twine',
    'twint',
    'absl-py',
    'tensorflow-datasets',
    'transformers',
    'tensorflow',
    'gensim',
    'scikit-learn',
    'google-cloud-storage',
    'mlflow',
    'termcolor',
    'memoized-property',
    'matplotlib',
    'nltk',
    'seaborn',
    'wordcloud',
    'psutil',
    'gcsfs',
    'streamlit',
    'pytz',
    'datetime',
    'altair',
    'plotly',
    'vega-datasets'
]

setup(name='green_mood_tracker',
      version="1.0",
      description="Green Mood Tracker",
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      test_suite='tests',
      include_package_data=True,
      scripts=['scripts/green_mood_tracker-run'],
      zip_safe=False)
