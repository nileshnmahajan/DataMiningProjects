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

from sklearn.metrics import v_measure_score
import numpy as np

with open('../data/true_labels.txt', 'r') as true:
    with open('../data/true_labels_num.txt', 'w') as iris_true:
        for line in true:
            if 'Iris-setosa' in line:
                iris_true.write("1" + "\n")
            if 'Iris-versicolor' in line:
                iris_true.write("2" + "\n")
            if 'Iris-virginica' in line:
                iris_true.write("3" + "\n")


true_labels = np.genfromtxt('../data/true_labels_num.txt')
print true_labels

iris_pred = np.genfromtxt('../predictions/kmeans_iris_1.txt')
print iris_pred

print v_measure_score(true_labels, iris_pred)
