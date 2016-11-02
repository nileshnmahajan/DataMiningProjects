import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from itertools import izip

train_df = pd.read_csv('../data/train.dat', sep=' ')

print train_df.shape


def get_unique_users_movies(dataset):
    users = set()
    movies = set()
    data = []
    y = []

    if dataset == "test":
        with open('../data/test_2.dat') as f:
            for line in f:
                (userID, movieID) = line.split(' ')
                data.append({"userID": str(userID), "movieID": str(movieID)})
                users.add(userID)
                movies.add(movieID)
        return (data, users, movies)

    if dataset == "train":
        with open('../data/train_2.dat') as f:
            for line in f:
                (userID, movieID, rating) = line.split(' ')
                data.append({"userID": str(userID), "movieID": str(movieID)})
                try:
                    # for matrix factorization, this was
                    y.append(float(rating))
                    # y.append(float(rating))
                except ValueError:
                    print "Check line {l}".format(l=line)
                users.add(userID)
                movies.add(movieID)
        return (data, y, users, movies)


train = get_unique_users_movies("train")
test = get_unique_users_movies("test")

X_train, y_train = train[0], train[1]

X_test = test[0]

print type(y_train)

v = DictVectorizer()
X_train_dv = v.fit_transform(X_train)
X_test_dv = v.transform(X_test)

print X_train_dv