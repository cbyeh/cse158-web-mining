import gzip
from collections import defaultdict
import ast
import random


def read_JSON(path):
    for l in gzip.open(path, 'rt', encoding="utf8"):
        d = ast.literal_eval(l)
        u = d['userID']
        try:
            g = d['gameID']
        except Exception as _:
            g = None
        yield u, g, d


# The goal is to rank which games are popular, and predict whether a user played a game
# An example line in train.json.gz representing a Steam game review
# {'hours': 0.3, 'gameID': 'b96045472', 'hours_transformed': 0.37851162325372983,
# 'early_access': False, 'date': '2015-04-08', 'text': '+1', 'userID': 'u01561183'}


data = []  # User, game played, and prediction
users = set()
games = set()
user_played = {}  # Dictionary of user and array of games played
game_users = {}  # Dictionary of games and users that play it

# Parse data
print("Reading data...")
for user, game, d in read_JSON("../PA3/data/train.json.gz"):
    data.append((user, game, 1))
    users.add(user)
    games.add(game)
print("Done")

# Frame training data
training_data = data[:]
for user, game, _ in training_data:
    if user not in user_played:
        user_played[user] = []
    user_played[user].append(game)
    if game not in game_users:
        game_users[game] = []
    game_users[game].append(user)


# Train using baseline method, threshold = top 66%
game_count = defaultdict(int)
total_played = 0
for user, game, _ in training_data:
    game_count[game] += 1
    total_played += 1
most_popular = [(game_count[x], x) for x in game_count]
most_popular.sort()
most_popular.reverse()
popular = set()
count = 0
for ic, i in most_popular:
    count += ic
    popular.add(i)
    if count > total_played * 0.65:
        break


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


print("Writing to predictions_Played.txt")
predictions = open("predictions/predictions_Played.txt", 'w')
for l in open("../PA3/data/pairs_Played.txt"):
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    user, game = l.strip().split('-')
    # From parsing data, games user played in training
    if (user not in user_played):
        pred = 0
        predictions.write(user + '-' + game + "," + str(pred) + "\n")
        continue
    games = user_played[user]  # g'
    if (game not in game_users):
        pred = 0
        predictions.write(user + '-' + game + "," + str(pred) + "\n")
        continue
    users_vali = game_users[game]
    if not games or not users_vali:  # Check if empty sets
        pred = 0
        predictions.write(user + '-' + game + "," + str(pred) + "\n")
        continue
    max = 0
    for g in games:
        users_train = game_users[g]
        j = jaccard(users_vali, users_train)
        if j > max:
            max = j
    if max > .045 or game in popular:
        pred = 1
    else:
        pred = 0
    predictions.write(user + '-' + game + ',' + str(pred) + '\n')
predictions.close()
# Kaggle username: christopheryeh
