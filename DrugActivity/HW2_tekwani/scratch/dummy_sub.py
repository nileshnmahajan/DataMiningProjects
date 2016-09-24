import csv
import pandas as pd

# choices = ["0", "1"]
# with open('dummy_sub.txt', 'w') as f:
#     for i in range(0, 350):
#         f.write("1" + "\n")


# with open("../data/molecule1.txt", 'r') as f:
#    line = f.readline()
#    x = line.split(" ")
#
#
# # print len(z)
# print len(x)

df_train = pd.read_csv('../data/train.csv', sep='\t', index_col=False, names=['Active', 'Structure'],
                       header=None)


