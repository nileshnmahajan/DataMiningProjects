from sgd import X_train, X_test, y_train
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from time import time
from imblearn.over_sampling import SMOTE

nb = BernoulliNB()

ratio ='auto'

smote = SMOTE(ratio=ratio, kind='regular')
smox, smoy = smote.fit_sample(X_train.toarray(), y_train)

print "Smote X_train ",smox




#
# nb.fit(smox, smoy)
# start = time()
# y_pred = nb.predict(X_test)
# print "Predicted values" , y_pred
# np.savetxt('../predictions/nb_predictions.txt', y_pred, fmt='%i')

print ("Finished classifying 350 drugs in: ", (time() - start))