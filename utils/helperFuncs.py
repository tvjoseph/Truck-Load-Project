from collections import Counter


# Method to get the most frequent item from a list
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]
