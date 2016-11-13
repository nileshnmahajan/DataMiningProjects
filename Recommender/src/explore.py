import pandas as pd
import graphlab as gl

actors_df = pd.read_csv('../data/movie_actors.dat', sep='\t')
train_df = pd.read_csv('../data/train.dat', sep=' ')
test_df = pd.read_csv('../data/test.dat', sep=' ')

train_sf = gl.SFrame(train_df)

unique_actors = list(actors_df['actorID'].unique())

# print len(unique_actors)

print len(train_df['user_id'].unique())

print len(test_df['user_id'].unique())

# Both train and test have the same number of users - 2113

# There are no new users
print "Any new users?: "
print test_df[~test_df['user_id'].isin(train_df['user_id'])]


print "Any new movies? :"

print test_df[~test_df['movieID'].isin(train_df['movieID'])]



# Prepare the data by removing items that are rare

rare_movies = train_sf.groupby('movieID', gl.aggregate.COUNT).sort('Count')
rare_movies = rare_movies[rare_movies['Count'] <= 3]
train_sf = train_sf.filter_by(rare_movies['movieID'], 'movieID', exclude=True)

rare_users = train_sf.groupby('user_id', gl.aggregate.COUNT).sort('Count')
rare_users = rare_users[rare_users['Count'] <= 15]
train_sf = train_sf.filter_by(rare_users['user_id'], 'user_id', exclude=True)


print "Original", train_df.shape
print "Reduced" , train_sf.shape

