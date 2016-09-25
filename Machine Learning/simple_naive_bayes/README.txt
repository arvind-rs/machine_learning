README:
This is a simple python implementation of the Naive Bayes Classifier.

Experimental Setup:
Hardware: Intel Core i7 6700HQ CPU, 2.6Ghz, 16GB RAM
Softare: Ubuntu 16.04, Python 2.7

Initial steps:
1. Extract the files to a folder.

Preprocessing steps:
1. Install virtualenv on your linux machine.
pip install virtualenv
2. Create a virtualenv in the folder with the extracted files
virtualenv nltk
3. Activate the virtualenv.
source nltk/bin/activate
4. Install nltk.
pip install nltk
import nltk
nltk.download()
5. Run the preprocessor.py file.
python preprocessor.py

Classifier steps:
1. Run the classifier.
python simple_naive_bayes_classifier.py
