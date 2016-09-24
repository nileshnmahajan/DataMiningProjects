from nltk.corpus import stopwords
import os
import string

stop = stopwords.words('english')

os.chdir('../')
output = os.getcwd() + '/data/stopwordsout.csv'
input = os.getcwd() +'/data/train.csv'


#for text files
def remove_stopwords(input, output):
    with open(input,'r') as inFile, open(output,'w') as outFile:
        for line in inFile:
            print >> outFile, (' '.join([word for word in line.lower().translate(None,'.\/?!').split()
                if len(word) >= 4 and word not in stop]))


#for DataFrame
remove_stopwords_df = lambda (x): (' '.join([word for word in x.lower().translate(None,'.\/?!').split()
            if len(word) >= 4 and x != "" and word not in stop]))



remove_stopwords(input=input, output=output)
