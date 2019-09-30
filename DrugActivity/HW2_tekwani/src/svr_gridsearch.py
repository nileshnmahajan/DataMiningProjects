import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

df_train = pd.read_csv('../data/train.csv', sep='\t', index_col=False, header=None,
                       names=['Active', 'Structure'])

vec = CountVectorizer(binary=True, vocabulary=[str(i) for i in range(100000)])
X_train = vec.fit_transform(df_train['Structure'])
y_train = df_train['Active'].values


param_grid = [{'C': 0.01}, {'C': 0.1}, {'C': 1.0}, {'C': 10.0}, {'C': 100.0}, {'C': 1000.0}, {'C': 10000.0},
              {'kernel': 'linear'}, {'kernel': 'rbf'}, {'gamma': 'auto'}, {'tol': 0.001},
              {'class_weight': {1:10}}, {'random_state': 456}]

estimator = svm.SVC()
selector = RFECV(estimator, step=1, cv=4)
clf = GridSearchCV(selector, {'estimator_params': param_grid}, cv=7)
clf.fit(X_train.toarray(), y_train)

print(clf.best_estimator_.estimator_)
print(clf.best_estimator_.grid_scores_)
print(clf.best_estimator_.ranking_)