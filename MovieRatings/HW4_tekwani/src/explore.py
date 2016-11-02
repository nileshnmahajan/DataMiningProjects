import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

train_df = pd.read_csv('../data/train.dat', sep=' ')
test_df = pd.read_csv('../data/test.dat', sep=' ')
movie_genres = pd.read_csv('../data/movie_genres.dat', sep= '\t')

genres = list(movie_genres['genre'].unique())

ratings = list(train_df['rating'].unique())

print ratings

movie_genres = pd.pivot_table(movie_genres, index=['movieID'], aggfunc= lambda x:', '.join(x))
genre_train = pd.DataFrame.join(train_df, movie_genres, on='movieID', lsuffix='_train', rsuffix='_gen', how='inner')

# One Hot Encoding for genres
for g in genres:
    genre_train[str(g)] = genre_train['genre'].str.contains(g)


genre_train = genre_train.drop('genre', 1)

print genre_train

genre_train.to_csv('genre_train.csv', sep=',', header=True)