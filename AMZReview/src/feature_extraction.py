import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from stopwords import remove_stopwords_df
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from time import time
import numpy as np
from nltk.stem import WordNetLemmatizer
import os


os.chdir('../HW1_tekwani')

csv_file = os.getcwd() + "/data/train.csv"
test_path = os.getcwd() + "/data/test.csv"

wordnet_lemma = WordNetLemmatizer()

# Getting train and test data into DataFrames.

df_test = pd.read_csv(test_path, lineterminator="\n", header = None, names=['Review'],
                      sep='\n', skip_blank_lines=False, warn_bad_lines=True)


df = pd.read_csv(csv_file, sep='\t', header=None, names=['score', 'review'], skip_blank_lines=False)
df.review = df.review.astype(str)
df.score = df.score.astype(str)
reviews = df.review
scores = df.score

df_test.Review = df_test.Review.astype(str)

smallcase = lambda(x): x.lower() if type(x) is str and not "" else x
lemmatizer = lambda (x): ",".join([wordnet_lemma.lemmatize(kw) for kw in x.split(" ")]) if x != "" else x

# Text preprocessing - removing numbers, keeping letters only

df['LettersOnly'] = df['review'].replace('[^A-Za-z]', ' ', regex=True)
df['LowerCase'] = df['LettersOnly'].apply(smallcase)
df['NoStopWords'] = df['LowerCase'].apply(remove_stopwords_df)
df['Lemmatized'] = df['NoStopWords'].apply(lemmatizer)

train_list = list(df.Lemmatized.values)

train_vocabulary = []

for review in train_list:
    train_vocabulary.extend(review.split())

trainvocabulary = set(train_vocabulary)

print "Train vocab size: ", len(trainvocabulary)

df_test['LettersOnly'] = df_test['Review'].replace('[^A-Za-z]', ' ', regex=True)
df_test['LowerCase'] = df_test['LettersOnly'].apply(smallcase)
df_test['NoStopWords'] = df_test['LowerCase'].apply(remove_stopwords_df)
df_test['Lemmatized'] = df_test['NoStopWords'].apply(lemmatizer)
test_list = list(df_test.Lemmatized.values)

# print test_list

test_vocabulary = []

for review in test_list:
    test_vocabulary.extend(review.split())

testvocabulary = set(test_vocabulary)

print "Test vocab size: ", len(testvocabulary)

# print "Test vocab", testvocabulary

vocabulary = testvocabulary | trainvocabulary

# print "Corpus vocab", vocabulary
print "Total length of vocabulary: ", len(vocabulary)


# Vectorizing Train data

vect = TfidfVectorizer(sublinear_tf=True, min_df=0.5, vocabulary=vocabulary, analyzer='word')
X = vect.fit_transform(train_list)
vocab = vect.get_feature_names()
# train_matrix = X.todense()
train_idf = vect.idf_


# Vectorizing test data
vect_test = TfidfVectorizer(sublinear_tf=True, min_df=0.5, vocabulary=vocabulary, analyzer='word')
Y = vect_test.fit_transform(test_list)
vocab_test = vect.get_feature_names()
test_idf = vect_test.idf_

start = time()

print ("Creating an LSA pipeline...")

svd = TruncatedSVD(n_components=2, algorithm='randomized')
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
lsa_train = lsa.fit_transform(X)
lsa_test = lsa.fit_transform(Y)


print("Latent Semantic Analysis done in %fs" % (time() - start))

train_linear_comb = pd.DataFrame(lsa_train, index=train_list, columns=["Component1", "Component2"])
test_linear_comb = pd.DataFrame(lsa_test, index=test_list, columns=["Component1", "Component2"])


print ("Writing LSA results to file")
train_linear_comb.to_csv('Train_LSA.csv', sep=',', columns=["Component1", "Component2"])
test_linear_comb.to_csv('Test_LSA.csv', sep=',', columns=["Component1", "Component2"])


print "Sample of components from train set"
print train_linear_comb.head(10)

print "Sample of components from test set"
print test_linear_comb.head(10)

print ("Computing similarity measures between train and test data...")

start_similarity_comp = time()

similarity = np.asarray(np.asmatrix(train_linear_comb) * np.asmatrix(test_linear_comb.T))

print pd.DataFrame(similarity, index=test_list, columns=train_list).head(10)

similarity_measures_df = pd.DataFrame(similarity)

print ("Similarity measures computed in %fs" %(time() - start_similarity_comp))
print similarity_measures_df.info()
print similarity_measures_df.head(10)

print ("Writing similarity measures to disk...")
t = time()

# similarity measures of train and test data split into chunks of 974 (18506/19) reviews (in rows) each


def save_df(similarity_measures_df, chunk_size=974):
    for i, start in enumerate(range(0, similarity_measures_df.shape[0], chunk_size)):
        similarity_measures_df[start:start+chunk_size].to_csv('df_{}.csv'.format(i))

save_df(similarity_measures_df, chunk_size=974)



print "Wrote file in: ", (time() - t)
