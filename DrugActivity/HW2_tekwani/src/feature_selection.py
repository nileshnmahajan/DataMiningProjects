import numpy as np
from scipy.sparse import csr_matrix
from exploring_data import X_train, X_test, selector, featurespace_dense_test, featurespace_dense_train, Y_train
from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt
from sklearn.metrics import accuracy_score, classification_report


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

F = load_sparse_csr('features.npz')
v = selector.fit(X_train)

var = v.variances_

# print "Features with highest variance", np.argsort(var)[-10:]

# thresh_0 = X[:, var > 0.03]

# print "thresh0: ", thresh_0
#
# # tells us how many features are left
# print thresh_0.shape

#feature numbers for the ones that are left
idx = np.where(v.variances_ > 0.14)[0]

# print "Feature numbers with highest variance" , idx
# print idx.shape
# print "no of features passing through current variance filter", len(idx)

# print type(idx)

# print "The features which are above the threshold: ", selector.get_support(indices=True)
# #No of features being used
# used = str(Counter(selector.get_support())[True])
# discarded = str(Counter(selector.get_support())[False])
# print "Using: ", used
# print "Discarded: ", discarded


df_reduced_train = pd.DataFrame(np.nan, index=range(800), columns=idx)
df_reduced_test = pd.DataFrame(np.nan, index=range(350), columns=idx)


def get_value_featurespace(row, column, test_train):
    if test_train == "test":
        return featurespace_dense_test[row, column]
    else:
        return featurespace_dense_train[row, column]


def populate_df_reduced(row, col, test_train):
    if test_train == "test":
        df_reduced_test.xs(row)[col] = get_value_featurespace(row, col, "test")
    else:
        df_reduced_train.xs(row)[col] = get_value_featurespace(row, col, "train")


def create_new_featurespace():
    for i in range(800):
        for j in idx:
            populate_df_reduced(i, j, "train")
    for i in range(350):
        for j in idx:
            populate_df_reduced(i, j, "test")


# Building the new test and train feature spaces
create_new_featurespace()


# result = pd.concat([df_reduced, Y], axis=1)
# new_pos = result['Active']
# result.drop(labels=['Active'], axis=1,inplace=True)
# result.insert(0, 'Active', new_pos)
# print result

print "New shape of train set", df_reduced_train.shape
print "New shape of test set", df_reduced_test.shape

train_target = Y_train.values

print "New test set: ", df_reduced_test

print "-----------------------------------------------"

print "New train set: ", df_reduced_train

clf = RandomForestClassifier(n_estimators=100)

#Fitting RandomForestClassifier on train data

clf.fit(df_reduced_train.values, train_target)

#Predicting for test data
Z = clf.predict(df_reduced_test.values)
savetxt('rf_predictions.txt', Z, fmt='%i')


print "Predictions for test: ", Z

#
# # print "Accuracy = %0.2f" % (accuracy_score(train_target.values, Z))
#
# # print ("Classification report: ")
#
# # print (classification_report(Y.values, Z, target_names=['Inactive', 'Active']))
#
#
