import graphlab as gl
from data_prep import user_tags_sf, movies_sf, train_sf_red

# Model 3 : reduced train set and side data : user and movie both

user_tags_sf.show()

params = dict([('target', 'rating'),
                   ('num_factors', [8, 10, 32, 48, 64]),
                   ('regularization', [.001, 0.0001, 0.00001, 0.0000001]),
                   ('max_iterations', [50, 100, 250]),
                   ('user_id', 'user_id'),
                   ('item_id', 'movieID'),
                ('item_data', movies_sf),
               ('user_data', user_tags_sf)
               ])

training_data, test_data = train_sf_red.random_split(0.5, seed=5)

job = gl.grid_search.create((training_data, test_data), gl.recommender.factorization_recommender.create,params,
                            return_model=True)
job.get_results()
job.get_best_params()

print job.get_metrics()

gl.show()

input()

