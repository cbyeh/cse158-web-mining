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
for user, game, d in read_JSON("data/train.json.gz"):
    data.append((user, game, 1))
    users.add(user)
    games.add(game)
print("Done")

# Split training and validation data
training_data = data[:165000]
for user, game, _ in training_data:
    if user not in user_played:
        user_played[user] = []
    user_played[user].append(game)
    if game not in game_users:
        game_users[game] = []
    game_users[game].append(user)
validation_data = data[165000:175000]


"""Begin question 1"""


# Add negatives to validation data
validation_data_copy = list.copy(validation_data)
for d in validation_data_copy:
    user = d[0]
    user_games_played = user_played[user]
    new_game = None
    while True:
        new_game = random.choice(tuple(games))
        if new_game not in user_games_played:
            break
    validation_data.append((user, new_game, 0))

# Train using baseline method
game_count = defaultdict(int)
total_played = 0
for user, game, _ in training_data:
    game_count[game] += 1
    total_played += 1
most_popular = [(game_count[x], x) for x in game_count]
most_popular.sort()
most_popular.reverse()
return1 = set()
count = 0
for ic, i in most_popular:
    count += ic
    return1.add(i)
    if count > total_played / 2:
        break

# Predict using baseline method
correct = 0
for user, game, pred in validation_data:
    if game in return1:
        if pred == 1:
            correct += 1
    elif pred == 0:
        correct += 1

accuracy = correct / len(validation_data)
print("Q1 Accuracy: " + str(accuracy))
# Q1 Accuracy: 0.68125


"""Begin question 2"""


# Train using baseline method, threshold = top 66%
game_count = defaultdict(int)
total_played = 0
for user, game, _ in training_data:
    game_count[game] += 1
    total_played += 1
most_popular = [(game_count[x], x) for x in game_count]
most_popular.sort()
most_popular.reverse()
return2 = set()
count = 0
for ic, i in most_popular:
    count += ic
    return2.add(i)
    if count > total_played / 1.5:  # 66th percentile
        break

# Predict using baseline method
correct = 0
for user, game, pred in validation_data:
    if game in return2:
        if pred == 1:
            correct += 1
    elif pred == 0:
        correct += 1

accuracy = correct / len(validation_data)
print("Q2 Accuracy: " + str(accuracy))
# Q2 Accuracy: 0.7007


"""Begin question 3"""


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# Predict using Jaccard similarity, threshold distance = 0.029
correct = 0
for user, game, pred in validation_data:
    # From parsing data, games user played in training
    games = user_played[user]  # g'
    users_vali = game_users[game]
    if not games or not users_vali:  # Check if empty sets
        if pred == 0:
            correct += 1
        continue
    max = 0
    for g in games:
        users_train = game_users[g]
        j = jaccard(users_vali, users_train)
        if j > max:
            max = j
    if max > .029:
        if pred == 1:
            correct += 1
    elif pred == 0:
        correct += 1

accuracy = correct / len(validation_data)
print("Q3 Accuracy: " + str(accuracy))
# Q3 Accuracy: 0.66795


"""Begin question 4"""


# Predict using Jaccard similarity, threshold distance = 0.029
correct = 0
for user, game, pred in validation_data:
    # From parsing data, games user played in training
    games = user_played[user]  # g'
    users_vali = game_users[game]
    if not games or not users_vali:  # Check if empty sets
        if pred == 0:
            correct += 1
        continue
    max = 0
    for g in games:
        users_train = game_users[g]
        j = jaccard(users_vali, users_train)
        if j > max:
            max = j
    if max > .029 or game in return2:
        if pred == 1:
            correct += 1
    elif pred == 0:
        correct += 1

accuracy = correct / len(validation_data)
print("Q4 Accuracy: " + str(accuracy))
# Q4 Accuracy: 0.67485


"""Begin question 5"""


print("Writing to predictions_Played.txt")
predictions = open("data/predictions_Played.txt", 'w')
for l in open("data/pairs_Played.txt"):
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
    if max > .04 or game in return2:
        pred = 1
    else:
        pred = 0
    predictions.write(user + '-' + game + "," + str(pred) + "\n")
predictions.close()
# Kaggle username: Christopher Yeh
