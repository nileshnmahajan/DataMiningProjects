from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
import numpy as np

TRUE_LABELS = '../data/true_labels.txt'
TRUE_LABELS_NUM = '../data/true_labels_num.txt'
PRED_PATH = '../predictions/kmeans_iris_3.txt'

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
