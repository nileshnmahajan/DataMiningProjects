# with open('../predictions/sklearn_benchmark.txt', 'r') as old:
#     with open('../predictions/sklearn_bench_pred.txt', 'w') as new:
#         for line in old:
#             if "0" in line:
#                 new.write("1" + "\n")
#             if "1" in line:
#                 new.write("2" + "\n")
#             if "2" in line:
#                 new.write("3"+ "\n")
#             if "3" in line:
#                 new.write("4"+ "\n")
#             if "4" in line:
#                 new.write(("5" + "\n"))
#             if "5" in line:
#                 new.write("6" + "\n")
#             if "6" in line:
#                 new.write("7"+ "\n")

from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score, silhouette_score
import numpy as np

TRUE_LABELS = '../data/true_labels.txt'
TRUE_LABELS_NUM = '../data/true_labels_num.txt'
PRED_PATH = '../predictions/kmeans_iris_2.txt'

with open(TRUE_LABELS, 'r') as true:
    with open(TRUE_LABELS_NUM, 'w') as iris_true:
        for line in true:
            if 'Iris-setosa' in line:
                iris_true.write("1" + "\n")
            if 'Iris-versicolor' in line:
                iris_true.write("2" + "\n")
            if 'Iris-virginica' in line:
                iris_true.write("3" + "\n")


true_labels = np.genfromtxt(TRUE_LABELS_NUM)
iris_pred = np.genfromtxt(PRED_PATH)

print "V-score", v_measure_score(true_labels, iris_pred)
print "Homogeneity score", homogeneity_score(true_labels, iris_pred)
print "Completeness score", completeness_score(true_labels, iris_pred)
