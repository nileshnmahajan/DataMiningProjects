from sklearn.linear_model import SGDClassifier
from imblearn.under_sampling import OneSidedSelection, NearMiss
from imblearn.ensemble import BalanceCascade
from collections import Counter
from feature_creation import df_reduced_train, y_train, df_reduced_test
from time import time
from numpy import savetxt

clf_start = time()

clf = SGDClassifier(n_iter=10000, loss='modified_huber', penalty='elasticnet', shuffle=True,
                    alpha=0.07)

clf.fit(df_reduced_train.values, y_train)
print "Time to run 10000 iterations of SGDClassifier", (time() - clf_start)
y_pred = clf.predict(df_reduced_test.values)
print "Predicted values: ", y_pred
savetxt('../predictions/sgd_predictions_16_vt_5.txt', y_pred, fmt='%i')
print ("Finished classifying 350 drugs in: ", (time() - clf_start ))

