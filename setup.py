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
      'google-cloud-storage==1.26.0',
      'mlflow==1.8.0',
      'termcolor==1.1.0',
      'memoized-property==1.0.3',
      'flask==1.1.1',
      'flask-cors',
      'gunicorn',
      'matplotlib',
      'nltk',
      'seaborn',
      'wordcloud',
      'psutil'
    ]

setup(name='green_mood_tracker',
      version="1.0",
      description="Green Mood Tracker",
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      test_suite = 'tests',
      include_package_data=True,
      scripts=['scripts/green_mood_tracker-run'],
      zip_safe=False)
