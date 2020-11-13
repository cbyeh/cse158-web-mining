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
for d in read_JSON("data/train_Category.json.gz"):
    data.append(d)
print("Done")


"""Begin question 6"""


training_data = data[:165000]
validation_data = data[165000:175000]

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

# Top 1000 words sorted
top_1000_words = dict(sorted(
    word_count.items(), key=itemgetter(1), reverse=True)[:1000])

# Find top 10 words and their frequencies
top_10_words = dict(sorted(
    word_count.items(), key=itemgetter(1), reverse=True)[:10])
top_10 = [(w, word_count[w] / total_words) for w in top_10_words]
print(top_10)
# [('the', 0.047446967093177694), ('and', 0.02767203214144606), ('a', 0.02660860784726279),
# ('to', 0.025429658351204455), ('game', 0.021376431377725155), ('of', 0.019797325582864286),
# ('is', 0.018157930617794103), ('you', 0.017479764576017718), ('i', 0.01707202856939985),
# ('it', 0.016637545777732472)]


"""Begin question 7"""


wordID = dict(zip(top_1000_words, range(len(top_1000_words))))


def count_freq_words(datum):
    feat = [0] * len(top_1000_words)
    t = datum['text']
    t = t.lower()
    t = [c for c in t if not (c in punct)]  # Non-punct characters
    t = ''.join(t)  # Convert back to string
    words = t.strip().split()  # Tokenizes
    for w in words:
        if w in top_1000_words:
            feat[wordID[w]] += 1
    feat.append(1)
    return feat


# Train data
x = [count_freq_words(d) for d in training_data]
y = [d['genreID'] for d in training_data]
model = linear_model.LogisticRegression(max_iter=10000)
model.fit(x, y)

# Evaluate data
x_validation = [count_freq_words(d) for d in validation_data]
y_validation = [d['genreID'] for d in validation_data]
predictions = model.predict(x_validation)
correct = 0
for i in range(len(y_validation)):
    if predictions[i] == y_validation[i]:
        correct += 1
accuracy = correct / len(y_validation)
print(accuracy)  # 0.6723


"""Begin question 8"""


# Try with top 1500 words
top_1500_words = dict(sorted(
    word_count.items(), key=itemgetter(1), reverse=True)[:1500])

wordID = dict(zip(top_1500_words, range(len(top_1500_words))))


def count_freq_words_more(datum):
    feat = [0] * len(top_1500_words)
    t = datum['text']
    t = t.lower()
    t = [c for c in t if not (c in punct)]  # Non-punct characters
    t = ''.join(t)  # Convert back to string
    words = t.strip().split()  # Tokenizes
    for w in words:
        if w in top_1500_words:
            feat[wordID[w]] += 1
    feat.append(1)
    return feat


# Try with more words
x = [count_freq_words_more(d) for d in training_data]
x_validation = [count_freq_words_more(d) for d in validation_data]
model = linear_model.LogisticRegression(max_iter=10000)
model.fit(x, y)
predictions = model.predict(x_validation)
correct = 0
for i in range(len(y_validation)):
    if predictions[i] == y_validation[i]:
        correct += 1
accuracy = correct / len(y_validation)
print(accuracy)  # 0.6927

# Write output
print("Writing to predictions_Category.txt")
output = open("data/predictions_Category.txt", 'w')
data = []
for d in read_JSON("data/test_Category.json.gz"):
    data.append(d)
x_test = [count_freq_words_more(d) for d in data]
model = linear_model.LogisticRegression(max_iter=10000)
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
