import gzip
from collections import defaultdict
import random
import re
import ast
import string
from operator import itemgetter
from sklearn import linear_model
import numpy as np


def read_JSON(path):
    for l in gzip.open(path, 'rt', encoding="utf8"):
        d = ast.literal_eval(l)
        yield d


# 'userID': 'u76283959', 'genre': 'Strategy', 'early_access': True, 'reviewID': 'r47002764', 'hours': 235.1,
# 'text': 'I think this is on my top 3 favorite TCGs of all time!', 'genreID': 1, 'date': '2017-05-04'}


data = []

# Parse data
print("Reading data...")
for d in read_JSON("../PA3/data/train_Category.json.gz"):
    data.append(d)
print("Done")


"""Begin question 6"""


training_data = data[:]

# Get rid of punctuation and capitalization in text and add to counts
word_count = defaultdict(int)
total_words = 0
punct = string.punctuation
for d in training_data:
    t = d['text']
    t = t.lower()  # Lowercase string
    t = [c for c in t if not (c in punct)]  # Non-punct characters
    t = ''.join(t)  # Convert back to string
    words = t.strip().split()  # Tokenizes
    for w in words:
        total_words += 1
        word_count[w] += 1


# Try with top 3500 words
top_3500_words = dict(sorted(
    word_count.items(), key=itemgetter(1), reverse=True)[:3500])
wordID = dict(zip(top_3500_words, range(len(top_3500_words))))


def count_freq_words_more(datum):
    feat = [0] * len(top_3500_words)
    t = datum['text']
    t = t.lower()
    t = [c for c in t if not (c in punct)]  # Non-punct characters
    t = ''.join(t)  # Convert back to string
    words = t.strip().split()  # Tokenizes
    for w in words:
        if w in top_3500_words:
            feat[wordID[w]] += 1
    feat.append(1)
    return feat


# Write output
print("Writing to predictions_Category.txt")
output = open("predictions/predictions_Category.txt", 'w')
data = []
for d in read_JSON("../PA3/data/test_Category.json.gz"):
    data.append(d)
x_test = [count_freq_words_more(d) for d in data]
model = linear_model.LogisticRegression(max_iter=10000)
x = [count_freq_words_more(d) for d in training_data]
y = [d['genreID'] for d in training_data]
model.fit(x, y)
predictions = model.predict(x_test)
output.write("userID-reviewID,prediction\n")
for i in range(len(x_test)):
    user = data[i]['userID']
    review = data[i]['reviewID']
    pred = predictions[i]
    output.write(user + '-' + review + "," + str(pred) + "\n")
output.close()
# Kaggle username: christopheryeh
