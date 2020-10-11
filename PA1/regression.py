import numpy as np
import json
import ast


def _parse_data(fname):
    """Load multiple JSON objects organized by line,
    by returning an eval each line
    """
    for l in open(fname):
        yield ast.literal_eval(l)


# Load dataset into a NumPy array
print("Reading data...")
data_list = list(_parse_data('data/fantasy_10000.json'))
print("Done")
data = np.array(data_list)

# Train a predictor to estimate rating from review length
for review in data:
    text = review['review_text']

print(data[0])
print(len(data))
