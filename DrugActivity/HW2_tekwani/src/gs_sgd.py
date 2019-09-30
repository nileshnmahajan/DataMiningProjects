from time import  time
import numpy as np
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn import grid_search
from sklearn.linear_model import SGDClassifier
from CV import X_train, y_train, X_test, y_test, df_reduced_train

clf = SGDClassifier()

parameters = [
                {'n_iter': [3000, 4000, 10000],
                 'penalty': ['l2', 'elasticnet'],
                 'loss': ['hinge', 'log', 'perceptron', 'modified_huber'],
                 'alpha': [0.03, 0.04, 0.07],
                 'shuffle': [True],
                 'class_weight': [{1:0.9}, {0: 0.1}, 'balanced']
                 }
            ]


start = time()
f1_scorer = make_scorer(f1_score)
gs = grid_search.GridSearchCV(clf, parameters, scoring=f1_scorer, n_jobs=-1)
gs.fit(df_reduced_train.values, y_train)

print("Grid scores: --------")
print(gs.grid_scores_)
print("Best estimator----")
print(gs.best_estimator_)
print("Best params ----")
print(gs.best_params_)
print("Best score: ", gs.best_score_)
print("Finished in: ", (time() - start))



