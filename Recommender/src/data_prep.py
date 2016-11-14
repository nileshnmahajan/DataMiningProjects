import pandas as pd
import graphlab as gl

train = pd.read_csv('../data/train.dat', sep=' ')
test = pd.read_csv('../data/test.dat', sep=' ')
actors_df = pd.read_csv('../data/movie_actors.dat', sep='\t')
usertags_df = pd.read_csv('../data/user_taggedmovies.dat', sep=' ')
tags_wt = pd.read_csv('../data/movie_tags.dat', sep='\t')
directors_df = pd.read_csv('../data/movie_directors.dat', sep='	')
top_actors = pd.read_csv('../data/top_actors.txt')
top_directors = pd.read_csv('../data/top_directors.txt')
movie_genres = pd.read_csv('../data/movie_genres.dat', sep='\t')



topdirectors = list(top_directors['directorID'].unique())
genres = list(movie_genres['genre'].unique())
movie_ids = list(train['movieID'].unique())
topactors = list(top_actors['actorID'].unique())

movie_genres = pd.pivot_table(movie_genres, index=['movieID'], aggfunc= lambda x:', '.join(x))
actors_new = pd.pivot_table(actors_df, index=['movieID'], aggfunc=lambda x: ', '.join(x))
directors_df = pd.pivot_table(directors_df, index=['movieID'], aggfunc= lambda x: ', '.join(x))

actors_new['movieID'] = actors_new.index

directors_SF = gl.SFrame(directors_df)
genres_SF = gl.SFrame(movie_genres)
tags_sf = gl.SFrame(tags_wt)
test_sf = gl.SFrame(test)
train_sf = gl.SFrame(train)
user_tags_sf = gl.SFrame(usertags_df)

# One Hot Encoding for genres and actors

for g in genres:
    movie_genres[str(g)] = movie_genres['genre'].str.contains(g)

for a in topactors:
    actors_new[str(a)] = actors_new['actorID'].str.contains(a)

for d in topdirectors:
    directors_df[str(d)] = directors_df['directorID'].str.contains(d)

directors_df = directors_df.drop('directorID', 1)
directors_df = directors_df.drop('directorName', 1)

movies_info = pd.DataFrame.join(movie_genres, actors_new, how='inner')
frames = [movies_info, directors_df]
x = pd.concat(frames, axis=1)
x.drop('genre', 1, inplace=True)

# x.to_csv('movies_sf.csv', sep=',')

movies_sf = gl.SFrame.read_csv('movies_sf.csv', delimiter=',', column_type_hints=[int,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str,str])
movies_sf.dropna()

# Rare user and movie filter
# train_sf_red is the reduced training set after filtering for rare users and movies

rare_movies = train_sf.groupby('movieID', gl.aggregate.COUNT).sort('Count')
rare_movies = rare_movies[rare_movies['Count'] <= 3]
train_sf_red = train_sf.filter_by(rare_movies['movieID'], 'movieID', exclude=True)

rare_users = train_sf.groupby('user_id', gl.aggregate.COUNT).sort('Count')
rare_users = rare_users[rare_users['Count'] <= 10]
train_sf_red = train_sf.filter_by(rare_users['user_id'], 'user_id', exclude=True)


