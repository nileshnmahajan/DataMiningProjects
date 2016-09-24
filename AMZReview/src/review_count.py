
def CountTrainReviews(location, character):
    subj = file(location, "r")
    count = 0
    for line in subj:
        count = count + line.count(character)
    return count

location = "/HW1_tekwani/data/train.csv"
character = "-1"
print CountTrainReviews(location, character)

character = "+1"
print CountTrainReviews(location, character)


def CountTestReviews(location, character):
    subj = file(location, "r")
    count = 0
    for line in subj:
        count = count + line.count(character)
    return count

location = "/home/bhavika/Desktop/CS584/HW1/datapipeline/train.data"
character = "\n"
print CountTestReviews(location, character)
