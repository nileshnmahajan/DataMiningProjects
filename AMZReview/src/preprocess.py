import unicodecsv as csv

txtfile = r"/home/bhavika/Desktop/CS584/HW1/datapipeline/train.txt"
csvtrain = r"/home/bhavika/Desktop/CS584/HW1/datapipeline/train.csv"

train_new = r"/home/bhavika/Desktop/CS584/HW1/datapipeline/train_new.csv"

in_txt = csv.reader(open(txtfile, "rb"), delimiter = '\t')
out_csv = csv.writer(open(csvtrain, "wb"))

out_csv.writerows(in_txt)

#adding opening and close quotes to reviews when there are none

score = "1,"
eol = "\n"

with open(csvtrain) as f:
    with open(train_new, "w") as fnew:
        for line in f:
            if score in line and '"' not in line:
                start = line.find(score)
                end = line.find(eol)
                newline = line[:start+2] + '"' + line[start+2:end-1] + '"' + line[end:]
                fnew.write(newline)
            else:
                fnew.write(line)
    f.close()

