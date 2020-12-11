import gzip
from collections import defaultdict
import scipy
import scipy.optimize
import numpy
import random

# From https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz
path = "C://Users/Julian McAuley/Documents/class_files/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"

f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')

dataset = []

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)

# First we'll build a few useful data structures, in this case just to maintain a collection of the items reviewed by each user, and the collection of users who have reviewed each item.
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)

itemNames = {}

for d in dataset:
    user,item = d['customer_id'], d['product_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    itemNames[item] = d['product_title']

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

def mostSimilar(i):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem: # For all items
        if i == i2: continue # other than the query
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:10]

query = dataset[2]['product_id']

# Retrieve most similar items
mostSimilar(query)

# and get their names
[itemNames[x[1]] for x in mostSimilar(query)]


# Efficient similarity-based recommendation

def mostSimilarFast(i):
    similarities = []
    users = usersPerItem[i]
    candidateItems = set()
    for u in users:
        candidateItems = candidateItems.union(itemsPerUser[u])
    for i2 in candidateItems:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:10]

mostSimilarFast(query)

reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    user,item = d['customer_id'], d['product_id']
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)

ratingMean = sum([d['star_rating'] for d in dataset]) / len(dataset)

# Our prediction function computes (a) a list of the user's previous ratings (ignoring the query item); and (b) a list of the similarities of these previous items, compared to the query. These weights are used to constructed a weighted average of the ratings from the first set.
def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['product_id']
        if i2 == item: continue
        ratings.append(d['star_rating'])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean

u,i = dataset[2]['customer_id'], dataset[2]['product_id']

predictRating(u, i)

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

alwaysPredictMean = [ratingMean for d in dataset]

cfPredictions = [predictRating(d['customer_id'], d['product_id']) for d in dataset]

labels = [d['star_rating'] for d in dataset]


MSE(alwaysPredictMean, labels)
MSE(cfPredictions, labels)

# In this case, the accuracy of our rating prediction model was actually _worse_ (in terms of the MSE) than just predicting the mean rating. However note again that this is just a heuristic, and could be modified to improve its predictions (e.g. by using a different similarity function other than the Jaccard similarity).
