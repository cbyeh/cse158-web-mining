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


# An example line in train.json.gz representing a Steam game review
# {'hours': 0.3, 'gameID': 'b96045472', 'hours_transformed': 0.37851162325372983,
# 'early_access': False, 'date': '2015-04-08', 'text': '+1', 'userID': 'u01561183'}


data = []
users = set()
games = set()
user_played = {}  # Dictionary of user and array of games played

# Parse data
for user, game, d in read_JSON("data/train.json.gz"):
    data.append(d)
    users.add(user)
    games.add(game)
    if user not in user_played:
        user_played[user] = []
    user_played[user].append(game)

print(user_played)
