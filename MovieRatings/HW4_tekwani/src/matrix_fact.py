from sklearn.metrics import mean_squared_error
import pickle
from sklearn.externals import joblib
from time import time
from pyfm import pylibfm
from sklearn.cross_validation import train_test_split
from feature_extraction import X_train_dv, X_test_dv, y_train


start_fact = time()
fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task='regression', initial_learning_rate=0.0001,
                learning_rate_schedule='optimal')
fm.fit(X_train_dv, y_train)

joblib.dump(fm, '../pickles/fm_100_7.pkl')

# fm = joblib.load('fm_30.pkl')
print "Finished fitting model in", time() - start_fact
preds = fm.predict(X_test_dv)
print preds

print type(preds)

joblib.dump(preds, '../pickles/predictions_6.pkl')



