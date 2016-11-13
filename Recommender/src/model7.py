import pandas as pd
import graphlab
from data_prep import movies_sf, train_sf, test_sf, user_tags_sf


# Based on observed RMSE from cv.py grid search

m7 = graphlab.factorization_recommender.create(train_sf, user_id='user_id', regularization= 0.0000001,
                                               max_iterations=50, num_factors=32,
                                               solver='adagrad',
                                               item_data=movies_sf, side_data_factorization=True,
                                               item_id='movieID', target='rating')


op = m7.predict(test_sf)

op.save('../predictions/model7_pred.txt', format='csv')


