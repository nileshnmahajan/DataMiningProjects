import numpy as np
from time import time
from scipy.sparse import csr_matrix
from feature_creation import X, y, featurespace_dense_X, selector
from collections import Counter
import pandas as pd
from sklearn.linear_model import SGDClassifier
from numpy import savetxt
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.cross_validation import StratifiedKFold
from imblearn.over_sampling import SMOTE

v = selector.fit(X)
start = time()

#feature numbers for the ones that are left
idx = np.where(v.variances_ > 0.04)[0]

df_reduced_train = pd.DataFrame(np.nan, index=range(800), columns=idx)


def get_value_featurespace(row, column, test_train):
    return featurespace_dense_X[row, column]


def populate_df_reduced(row, col, test_train):
    df_reduced_train.xs(row)[col] = get_value_featurespace(row, col, "train")


def create_new_featurespace():
    for i in range(800):
        for j in idx:
            populate_df_reduced(i, j, "train")


# Building the new test and train feature spaces
create_new_featurespace()

# print "New shape of train set", df_reduced_train.shape
# print "New train set: ", df_reduced_train


skf = StratifiedKFold(y, n_folds=15, shuffle=True)

for train_index, test_index in skf:
    # print ("TRAIN:", train_index, "TEST:", test_index)
    X_train, y_train = df_reduced_train.iloc[train_index], y[train_index]
    X_test, y_test = df_reduced_train.iloc[test_index], y[test_index]



print "X_test: ", X_test
print "X_test shape:", X_test.shape

print "Time elapsed: ", time() - start


