from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from feature_creation import X_train, y_train, df_reduced_test
from feature_creation import selector, idx, df_reduced_train
from time import time
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import numpy as np

smote = SMOTE(kind='borderline1', ratio='auto', k=10)

param_grid = [{'penalty': ['l2'],
               'C': [0.001, 0.01, 0.1, 1.0, 10, 100],
               'class_weight': ['balanced'],
               'max_iter': [100, 200, 500, 800, 1000],
               'solver': ['liblinear', 'newton-cg', 'lbfgs'],
               'multi_class': ['ovr'],
               'tol': [1e-4, 1e-3, 1e-2]
               }]


clf = LogisticRegression()
start = time()

f1_scorer = make_scorer(f1_score)

# Oversampling
X_tr, X_te, y_tr, y_te = train_test_split(df_reduced_train.values, y_train, test_size=0.3, stratify=y_train)
X_tr, y_tr = smote.fit_sample(X_tr, y_tr)

gs = GridSearchCV(clf, param_grid, scoring=f1_scorer, n_jobs=-1)
gs.fit(X_tr, y_tr)

y_pred = gs.predict(X_te)

print "Grid scores: --------"
print gs.grid_scores_
print "Best estimator----"
print gs.best_estimator_
print "Best params ----"
print gs.best_params_
print "Best score: ", gs.best_score_
print "Finished in: ", (time() - start)

print ("Classification report: ")

print classification_report(y_te, y_pred)


