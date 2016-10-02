from time import  time
import numpy as np
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from feature_creation import X_train, y_train
from sklearn.tree import DecisionTreeClassifier
from time import time
from sklearn.cross_validation import StratifiedKFold
from feature_creation import idx
import pandas as pd

start = time()

f1_scorer = make_scorer(f1_score)

parameters = [{'n_estimators': [50, 100, 500, 1000, 2500],
               'base_estimator__criterion': ["gini", "entropy"],
               'base_estimator__splitter': ["best", "random"],
               }]


df_reduced_train = pd.DataFrame(np.nan, index=range(800), columns=idx)

skf = StratifiedKFold(y_train, n_folds=5, shuffle=True)

for train_index, test_index in skf:
    # print ("TRAIN:", train_index, "TEST:", test_index)
    X_train, y_train_skf = df_reduced_train.iloc[train_index], y_train[train_index]
    X_test, y_test = df_reduced_train.iloc[test_index], y_train[test_index]


dtc = DecisionTreeClassifier(max_features="auto", class_weight="balanced",
                             max_depth=None)

ab = AdaBoostClassifier(base_estimator=dtc)

gs = GridSearchCV(ab, param_grid=parameters, scoring=f1_scorer)

gs.fit(X_test, y_train_skf)

print "Grid scores: --------"
print gs.grid_scores_
print "Best estimator----"
print gs.best_estimator_
print "Best params ----"
print gs.best_params_
print "Best score: ", gs.best_score_
print "Finished in: ", (time() - start)

y_true, y_pred = y_test, gs.predict(X_test)


print ("Classification report: ")

print classification_report(y_true, y_pred)





