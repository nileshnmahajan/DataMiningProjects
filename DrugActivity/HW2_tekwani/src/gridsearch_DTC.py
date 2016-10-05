from sklearn.tree import DecisionTreeClassifier
from feature_creation import X_train, X_test, y_train, df_reduced_train
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, make_scorer, roc_auc_score
from time import time

dt_start = time()

dt_clf = DecisionTreeClassifier()

parameters_grid = [
                {'criterion':['gini', 'entropy'],
                 'splitter': ['best', 'random'],
                 'max_features': ['auto', 'sqrt', 'log2'],
                 'max_depth': [2, 3, 4, 6, 8, 10],
                 'class_weight': [{1: 0.9, 0: 0.1}, 'balanced']
                 }
            ]

f1_scorer = make_scorer(f1_score)

grid_tree = GridSearchCV(dt_clf, param_grid=parameters_grid, scoring=f1_scorer)

grid_tree.fit(df_reduced_train.values, y_train)

print "Grid scores: --------"
print grid_tree.grid_scores_
print "Best estimator----"
print grid_tree.best_estimator_
print "Best params ----"
print grid_tree.best_params_
print "Best score: ", grid_tree.best_score_

print "Finished in: ", (time() - dt_start)



