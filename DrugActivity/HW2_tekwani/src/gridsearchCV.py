import numpy as np
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn import grid_search
from CV import X_train, y_train, clf, X_test, y_test
import subprocess
subprocess.call(['speech-dispatcher'])        #start speech dispatcher



parameters = [
                {'n_iter': [5, 20, 50, 100, 150, 175, 200, 250, 275, 300, 350, 1000, 2500],
                 'penalty': ['l2', 'elasticnet'],
                 'loss': ['hinge', 'log'],
                 'alpha': [0.001, 0.01, 0.02],
                 'shuffle': [True]}
            ]


f1_scorer = make_scorer(f1_score)

gs = grid_search.GridSearchCV(clf, parameters, scoring=f1_scorer, n_jobs=-1)
gs.fit(X_train, y_train)


print "Grid scores: --------"


print gs.grid_scores_


print "Best estimator----"

print gs.best_estimator_

print "Best params ----"

print gs.best_params_

print "Best score: ", gs.best_score_

subprocess.call(['spd-say', '"Your code has finished executing."'])
