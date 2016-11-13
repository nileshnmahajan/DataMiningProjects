import pandas as pd
import graphlab
from data_prep import movies_sf


# Based on observed RMSE from cv.py grid search

train_df = pd.read_csv('../data/train.dat', sep=' ')
test_df = pd.read_csv('../data/test.dat', sep=' ')
user_tags_df = pd.read_csv('../data/user_taggedmovies.dat', sep=' ')


train_SF = graphlab.SFrame(train_df)
X_test = graphlab.SFrame(test_df)
X_test_SF = graphlab.SFrame(X_test)


print movies_sf

user_tags_sf = graphlab.SFrame(user_tags_df)


m6 = graphlab.factorization_recommender.create(train_SF, user_id='user_id', regularization= 0.00001,
                                               max_iterations=100, num_factors=64,
                                               solver='adagrad', linear_regularization=0.00000001,
                                               item_data=movies_sf, side_data_factorization=True,
                                               item_id='movieID', target='rating')


op = m6.predict(X_test_SF)

op.save('../predictions/model6_pred.txt', format='csv')


