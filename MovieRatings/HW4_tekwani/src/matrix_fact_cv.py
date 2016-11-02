from sklearn.cross_validation import train_test_split
import pandas as pd
from time import time
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.externals import joblib

train_df = pd.read_csv('../data/train.dat', sep=' ')

print train_df.shape

def get_unique_users_movies():
    users = set()
    movies = set()
    data = []
    y = []

    with open('../data/train_2.dat') as f:
        for line in f:
            (userID, movieID, rating) = line.split(' ')
            data.append({"userID": str(userID), "movieID": str(movieID)})
            try:
                y.append(float(rating))
            except ValueError:
                print "Check line {l}".format(l = line)
            users.add(userID)
            movies.add(movieID)
    return (data, y, users, movies)


train = get_unique_users_movies()
X_train, X_test, y_train, y_test = train_test_split(train[0], train[1], test_size=0.3)

v = DictVectorizer()
X_train_dv = v.fit_transform(X_train)
X_test_dv = v.transform(X_test)


start_fact = time()
fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task='regression', initial_learning_rate=0.0001,
                learning_rate_schedule='optimal')
fm.fit(X_train_dv, y_train)
joblib.dump(fm, 'fm1.pkl')

# fm = joblib.load('fm_30.pkl')
print "Finished fitting model in", time() - start_fact

preds = fm.predict(X_test_dv)

print preds

print("FM MSE: %.4f" % mean_squared_error(y_test, preds))



