import pandas as pd
import graphlab
from data_prep import tags_sf, movies_sf, user_tags_sf, train_sf, test_sf



#Building a similarity model for users

# change num_factors to 8 for 0.77 result

model = graphlab.factorization_recommender.create(train_sf, user_id='user_id', item_id='movieID', target='rating', side_data_factorization=True,
                                              item_data=movies_sf, num_factors=64, max_iterations=100)


op = model.predict(test_sf)

op.save('../predictions/model1_pred.txt', format='csv')
