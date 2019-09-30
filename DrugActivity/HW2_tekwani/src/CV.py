import numpy as np
from time import time
from feature_creation import X_train, y_train, idx, featurespace_dense_X_train
import pandas as pd
from sklearn.linear_model import SGDClassifier
from numpy import savetxt
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt

start = time()

#feature numbers for the ones that are left

df_reduced_train = pd.DataFrame(np.nan, index=range(800), columns=idx)


def get_value_featurespace(row, column, test_train):
    return featurespace_dense_X_train[row, column]


def populate_df_reduced(row, col, test_train):
    df_reduced_train.xs(row)[col] = get_value_featurespace(row, col, "train")


def create_new_featurespace():
    for i in range(800):
        for j in idx:
            populate_df_reduced(i, j, "train")


# Building the new test and train feature spaces
create_new_featurespace()

skf = StratifiedKFold(y_train, n_folds=5, shuffle=True)

print("No of folds in Stratified K-Fold", skf.n_folds)

for train_index, test_index in skf:
    # print(("TRAIN:", train_index, "TEST:", test_index))
    X_train, y_true = df_reduced_train.iloc[train_index], y_train[train_index]
    X_test, y_test = df_reduced_train.iloc[test_index], y_train[test_index]


clf = SGDClassifier(n_iter=10000, alpha=0.07, loss='modified_huber', penalty='elasticnet', shuffle=True)
clf.fit(X_train, y_true)


Z = clf.predict(X_test)

print("Classified drugs from test set in : ", (time() - start))
print("Precision: ", precision_score(y_test, Z))
print("Recall: ", recall_score(y_test, Z))
print("F1 score: " , f1_score(y_test, Z,  average='binary'))

print("Classification report")

print("-------------------------------")

print(classification_report(y_test, Z, target_names=['Inactive', 'Active']))


print('auc', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

preds = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, preds)

roc = pd.DataFrame(dict(fpr=fpr,tpr=tpr))
roc_auc = auc(fpr, tpr)

plt.title("Receiver Operating Characteristics")
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])

plt.show()








