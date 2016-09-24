import pandas as pd
from time import time
import numpy as np
import os

os.chdir('../')
csv_file = os.getcwd() + "/data/train.csv"

sim_measure_i = pd.DataFrame()

df = pd.read_csv(csv_file, sep='\t', header=None, names=['score', 'review'], skip_blank_lines=False)
df.review = df.review.astype(str)
df.score = df.score.astype(str)
reviews = df.review
scores = df.score

df['train_case_id'] = df.index.values
df = df[['train_case_id', 'score', 'review']]
df = df.set_index('train_case_id')


def read_dataframe_part(i, k):
    df_load_time = time()
    sim_measure_i = pd.read_csv(os.getcwd()+'/df_{}.csv'.format(i),  index_col=0)
    print "Dataframe_{} read in: ".format(i), (time() - df_load_time)
    sim_measure_i.index.name = 'test_case_id'
    nearest_neighbours(sim_measure_i, k)


def get_review(train_case):
    return df.iloc[train_case].score


def nearest_neighbours(similarity_matrix, k):
    neighbours = pd.DataFrame(similarity_matrix.apply(lambda s: s.nlargest(k).index.tolist(), axis =1))
    neighbours = neighbours.rename(columns={0:'nbr_list'})
    neighbours['test_case_id'] = neighbours.index.values

    # getting the scores of each train neighbour from the train dataframe
    nbr_scores = []

    # creating a list of lists containing the sentiment scores for each set of train neighbours
    # [0, 1, 2, 3, 4] -> [1, 1, 1, 1, 1] from train data set

    test_scores_ref = []

    for i in range(neighbours.shape[0]): #row
        for j in range(len(neighbours['nbr_list'].iloc[i])):  #element
            a = neighbours['nbr_list'].iloc[i][j]
            nbr_scores.append(get_review(int(a)))

    test_scores_ref = [nbr_scores[i:i+k] for i in range(0, len(nbr_scores), k)]
    test_scores_series = pd.DataFrame(test_scores_ref)

    test_scores_series['Pos'] = test_scores_series.apply(lambda x: x.value_counts(), axis=1)['1']
    test_scores_series['Neg'] = test_scores_series.apply(lambda y: y.value_counts(), axis=1)['-1']
    test_scores_series['Pos'].fillna(0, inplace=True)
    test_scores_series['Neg'].fillna(0, inplace=True)

    # determining conditions for majority vote of nearest neighbours
    conditions = [(test_scores_series['Pos'] > test_scores_series['Neg']),
                  (test_scores_series['Pos'] < test_scores_series['Neg'])]

    # choices to be printed are +1 and -1
    choices = ["+1", "-1"]
    test_scores_series['Verdict'] = np.select(conditions, choices, default=np.nan)
    write_submission_file(test_scores_series)


def write_submission_file(test_scores_series):
    test_scores_series.to_csv('submission.txt', sep='\n', mode ='a',columns=['Verdict'], header=False, index=False)


def knn_classifier(k):
    start_classify = time()
    print "Classifying 18506 reviews using k=", k
    for i in range(0, 19):
        read_dataframe_part(i, k)
    end_classify = time() - start_classify

    print "Classified 18506 Amazon reviews in : ", end_classify

knn_classifier(379)

