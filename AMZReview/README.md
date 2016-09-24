## HW1: Amazon Review Classification

### Installation

1. Unzip HW1_tekwani.zip
2. cd/HW1_tekwani
3. If you want to create a virtual environment, run virtualenv <the name of the environment, say 'tekwani'>
4. To begin using the virtual environment, you must activate it.
   $ source tekwani/bin/activate
5. Now install the packages specified in requirements.txt. You can do this using
   pip freeze > requirements.txt (freeze the current state of the environment)
   pip install -r requirements.txt
6. Depending on your installation, NLTK might require WordNet and stopwords data. To install these, run python.

   >> import nltk
   >> nltk.download()

   When the graphical installer appears, select WordNet and stopwords from Corpora and install.

### Running the kNN classifier

The folder HW1_tekwani/data must contain a test.csv and train.csv file AS IT IS.
I've processed the test.data and train.data files for white spaces, formatting and quote line delimiters.
Some of the steps I've taken to process these files are in preprocess.py but I've used a combination of Unix commands and Python.
The classifier will not work on the train.data & test.data directly.

1. Run src/feature_extraction.py first. This usually takes about 20 minutes to generate 19 CSV files that contain similarity measures for train and test data.
2. Now run knn.py without any command line arguments. To change the value of k, edit the last line in knn.py `knn_classifier(379)`.

knn.py creates a submission.txt file that contains 18506 rows with either a +1 or a -1.

