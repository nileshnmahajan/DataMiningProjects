import numpy as np
from sklearn.externals import joblib

preds = joblib.load('../pickles/predictions_6.pkl')

print type(preds)

preds = np.around(preds, decimals=1)

print preds

np.savetxt("../predictions/trial6_rnd.txt", preds, delimiter='\n', fmt='%.2f')