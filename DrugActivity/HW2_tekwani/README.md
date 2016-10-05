## HW2: Drug Activity Prediction

### Installation

1. Unzip HW2_tekwani.zip
2. cd/HW2_tekwani
3. If you want to create a virtual environment, run virtualenv <the name of the environment, say 'tekwani'>
4. To begin using the virtual environment, you must activate it.
   $ source tekwani/bin/activate
5. Now install the packages specified in requirements.txt. You can do this using
   pip freeze > requirements.txt (freeze the current state of the environment)
   pip install -r requirements.txt


### Running the solution

The folder HW2_tekwani/data must contain a test.csv and train.csv file AS IT IS since the code references the `data` directory.

1. Run src/feature_creation.py.
2. Now run sgd.py without any command line arguments. It will generate a predictions.txt file in the `predictions` folder.
3. CV.py contains an implementation of`StratifiedKFold` - running it with 2 and 5 folds gives the best F1-score.
   Sample output for 2, 5 and 10 folds is in CV_2.out, CV_5.out and CV_10.out respectively.
    
sgd_predictions.txt contains 350 lines with 0 or 1 labels.

### Other files

1. `cv_sampling.py` : Compares the estimator performance of several classifiers combined with sampling techniques from imbalanced-learn. Output: cv_sampling.out
2. `gridsearch_adaboost.py` : Finds the best hyperparameter values for `AdaBoostClassifier` using `GridSearchCV`. Output: gs_ada.out
3. `gridsearch_DTC.py`: Find the best hyperparameter values for `DecisionTreeClassifier` using `GridSearchCV`. Output: not included
4. `gridseach_sgd.py`: Find the best hyperparametes for `SGDClassifier` using `GridSearchCV`. Takes very long to run. Output: not included
5. `smote_logreg.py`: Using SMOTE `borderline1` variant for oversampling and `GridSearchCV` fo find the best hyperparameters for `LogisticRegression`. Output: smote_logreg.out
6. `imbalances_sgd.py`: Testing to see which sampling technique helps boost `SGDClassifier` performance. The hyperparameter values for `SGDClassifier` are the optimal values selected using `gridsearch_SGD.py`. Output: imbalances_sgd.out

