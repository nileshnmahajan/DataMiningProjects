from sklearn.cross_validation import train_test_split
import pandas as pd
import graphlab

train_df = pd.read_csv('../data/train.dat', sep=' ')
test_df = pd.read_csv('../data/test.dat', sep=' ')
movie_genres = pd.read_csv('../data/movie_genres.dat', sep='\t')
actors_df = pd.read_csv('../data/movie_actors.dat', sep='\t')

actors_df = actors_df.drop('actorID', 1)

X = train_df.iloc[:, [0,1, 2]]

genres = list(movie_genres['genre'].unique())

ratings = list(train_df['rating'].unique())

# print ratings

movie_genres = pd.pivot_table(movie_genres, index='item_id', aggfunc= lambda x:', '.join(x))
genre_train = pd.DataFrame.join(train_df, movie_genres, on='item_id', lsuffix='_train', rsuffix='_gen', how='inner')

# One Hot Encoding for genres
for g in genres:
    genre_train[str(g)] = genre_train['genre'].str.contains(g)


genre_train = genre_train.drop('genre', 1)

# print genre_train
train_SF = graphlab.SFrame(train_df)
X_test = graphlab.SFrame(test_df)
X_test_SF = graphlab.SFrame(X_test)


#Building a similarity model for users

m = graphlab.factorization_recommender.create(train_SF, target='rating', side_data_factorization=False)
nn = m.get_similar_users()

# nn.export_csv('../scratch/similar_users.txt', delimiter=',')

op = m.predict(X_test_SF)

op.save('../scratch/pred1.txt', format='csv')