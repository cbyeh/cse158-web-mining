import gzip
import ast
from collections import defaultdict


def read_JSON(path):
    for l in gzip.open(path, 'rt', encoding="utf8"):
        d = eval(l)
        u = d['userID']
        try:
            g = d['gameID']
        except Exception as _:
            g = None
        yield u, g, d


# Time-played baseline: compute averages for each user, or return the global average if we've never seen the user before
all_hours = []
user_hours = defaultdict(list)

for user, game, d in read_JSON("data/train.json.gz"):
    h = d['hours_transformed']
    all_hours.append(h)
    user_hours[user].append(h)

global_average = sum(all_hours) / len(all_hours)
user_average = {}
for u in user_hours:
    user_average[u] = sum(user_hours[u]) / len(user_hours[u])

predictions = open("data/predictions_Hours.txt", 'w')
for l in open("data/pairs_Hours.txt"):
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    u, g = l.strip().split('-')
    if u in user_average:
        predictions.write(u + '-' + g + ',' + str(user_average[u]) + '\n')
    else:
        predictions.write(u + '-' + g + ',' + str(global_average) + '\n')

predictions.close()

# Would-play baseline: just rank which games are popular and which are not, and return '1' if a game is among the top-ranked

game_count = defaultdict(int)
total_played = 0

for user, game, _ in read_JSON("data/train.json.gz"):
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
    if count > total_played/2:
        break

predictions = open("data/predictions_Played.txt", 'w')
for l in open("data/pairs_Played.txt"):
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    u, g = l.strip().split('-')
    if g in return1:
        predictions.write(u + '-' + g + ",1\n")
    else:
        predictions.write(u + '-' + g + ",0\n")

predictions.close()

# Category prediction baseline: Just consider some of the most common words from each category

cat_dict = {
    "Action": 0,
    "Strategy": 1,
    "RPG": 2,
    "Adventure": 3,
    "Sport": 4
}

predictions = open("data/predictions_Category.txt", 'w')
predictions.write("userID-reviewID,prediction\n")
for u, _, d in read_JSON("data/test_Category.json.gz"):
    # If there's no evidence, just choose the most common category in the dataset
    cat = cat_dict['Action']
    words = d['text'].lower()
    if 'strategy' in words:
        cat = cat_dict['Strategy']
    if 'rpg' in words:
        cat = cat_dict['RPG']
    if 'adventure' in words:
        cat = cat_dict['Adventure']
    if 'sport' in words:
        cat = cat_dict['Sport']
    predictions.write(u + '-' + d['reviewID'] + "," + str(cat) + "\n")

predictions.close()
