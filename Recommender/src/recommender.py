import graphlab as gl
from data_prep import tags_sf, movies_sf, user_tags_sf, train_sf, test_sf


model = gl.factorization_recommender.create(train_sf,  side_data_factorization=True, target='rating', num_factors=48, user_id='user_id',
                                            regularization=0.0000001, item_id='movieID', item_data=movies_sf, linear_regularization=0.0000000001,
                                            max_iterations=50)


op = model.predict(test_sf)

op.save('../predictions/final_76_3.txt', format='csv')
