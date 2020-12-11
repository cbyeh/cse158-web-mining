##################################################
# Loading data from the web                      #
##################################################

from urllib.request import urlopen

# Grab the page from the web
f = urlopen("https://www.goodreads.com/book/show/4671.The_Great_Gatsby")
html = str(f.read())

# Extract the (30) review elements
reviews = html.split('<div id="review_')[1:]

# Parse the components of a single review
def parseReview(review):
    d = {}
    d['stars'] = review.split('<span class=" staticStars notranslate" title="')[1].split('"')[0]
    d['date'] = review.split('<a class="reviewDate')[1].split('>')[1].split('<')[0]
    d['user'] = review.split('<a title="')[1].split('"')[0]
    shelves = []
    try:
        shelfBlock = review.split('<div class="uitext greyText bookshelves">')[1].split('</div')[0]
        for s in shelfBlock.split('shelf=')[1:]:
            shelves.append(s.split('"')[0])
        d['shelves'] = shelves
    except Exception as e:
        pass
    reviewBlock = review.split('<div class="reviewText stacked">')[1].split('</div')[0]
    d['reviewBlock'] = reviewBlock
    return d

# Parse the list of reviews
reviewDict = [parseReview(r) for r in reviews]

### Parsing the data using BeautifulSoup

from bs4 import BeautifulSoup
soup = BeautifulSoup(reviewDict[0]['reviewBlock'])
print(soup.text)

### Using a headless browser (Note: requires some setup to install the library)

from splinter import Browser
browser = Browser("chrome")
browser.visit("https://www.goodreads.com/book/show/4671.The_Great_Gatsby")
html = browser.html


##################################################
# Manipulating time data                         #
##################################################

# Yelp challenge data: https://www.yelp.com/dataset/challenge

import time
import json

path = "datasets/yelp_data/review.json"
f = open(path, 'r', encoding = 'utf8')

# Read first 50000 entries
dataset = []
for i in range(50000):
    dataset.append(json.loads(f.readline()))

timeString = dataset[0]['date']

# String --> struct
timeStruct = time.strptime(timeString, "%Y-%m-%d")
time.strptime("21:36:18, 28/5/2019", "%H:%M:%S, %d/%m/%Y")

# Struct --> int
timeInt = time.mktime(timeStruct)
timeInt

timeInt2 = time.mktime(time.strptime(dataset[99]['date'], "%Y-%m-%d"))

timeDiff = timeInt - timeInt2

# Int --> struct
timeStruct2 = time.gmtime(timeInt2)

# Struct --> string
time.strftime("%b %Y, %I:%M:%S", timeStruct2)


##################################################
# Matplotlib                                     #
##################################################

datasetWithTimeValues = []

# Add time values to the dataset
for d in dataset:
    d['date']
    d['timeStruct'] = time.strptime(d['date'], "%Y-%m-%d")
    d['timeInt'] = time.mktime(d['timeStruct'])
    datasetWithTimeValues.append(d)

# Compile ratings per weekday
from collections import defaultdict
weekRatings = defaultdict(list)

for d in datasetWithTimeValues:
    day = d['timeStruct'].tm_wday
    weekRatings[day].append(d['stars'])

weekAverages = {}

for d in weekRatings:
    weekAverages[d] = sum(weekRatings[d]) / len(weekRatings[d])

# Plot...

X = [0,1,2,3,4,5,6]
Y = [weekAverages[x] for x in X]

import matplotlib.pyplot as plt

# Line plot
plt.plot(X, Y)
plt.show()

# Bar plot
plt.bar(X, Y)
plt.show()

# Limits
plt.bar(X, Y)
plt.ylim(3.6, 3.8)
plt.show()

# Label, ticks, and title
plt.ylim(3.6, 3.8)
plt.xlabel("Weekday")
plt.ylabel("Av. Rating")
plt.xticks(X, "SMTWTFS")
plt.title("weekday vs. rating")
plt.bar(X, Y)
plt.show()


##################################################
# Tensorflow                                     #
##################################################

import tensorflow as tf

# PM2.5 data: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
path = "datasets/PRSA_data_2010.1.1-2014.12.31.csv"
f = open(path, 'r')

dataset = []
header = f.readline().strip().split(',')
for line in f:
    line = line.split(',')
    dataset.append(line)

# Exclude N/A entries from the dataset
dataset = [d for d in dataset if d[5] != 'NA']

# Extract features
def feature(datum):
    feat = [1, float(datum[7]), float(datum[8]), float(datum[10])] # Temperature, pressure, and wind speed
    return feat

X = [feature(d) for d in dataset]
y = [float(d[5]) for d in dataset]

# Convert to tensorflow constant (column vector)
y = tf.constant(y, shape=[len(y),1])

K = len(X[0])

# Regularized MSE
def MSE(X, y, theta):
  return tf.reduce_mean((tf.matmul(X, theta) - y)**2) + 1.0 * tf.reduce_sum(theta**2)

theta = tf.Variable(tf.constant([0.0]*K, shape=[K,1]))

optimizer = tf.train.AdamOptimizer(0.01)
objective = MSE(X,y,theta)

train = optimizer.minimize(objective)

# Initialize variables

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Run gradient descent

for iteration in range(1000):
  cvalues = sess.run([train, objective])
  print("objective = " + str(cvalues[1]))

# Print results

with sess.as_default():
  print(MSE(X, y, theta).eval())
  print(theta.eval())
