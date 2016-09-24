import csv
import itertools

f1 = open("submission.txt")
f2 = open("/home/bhavika/PycharmProjects/AMZReview/HW1/datapipeline/train_scores.csv")

csv_f1 = csv.reader(f1)
csv_f2 = csv.reader(f2)

correct = 0
incorrect = 0


for row1, row2 in itertools.izip(csv_f1, csv_f2):
    if row1 == row2:
        correct += 1
    else:
        incorrect += 1

print "Correct: ", correct
print "Incorrect: ", incorrect

accuracy = float (correct)/(correct+incorrect)

print "Acc: ", accuracy