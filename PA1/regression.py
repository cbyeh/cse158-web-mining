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


def _text_length_feature(datum):
    """Return a vector of just one feature,
    The length of the review text in characters
    """
    feat = [1, len(datum['review_text'])]
    return feat


# Train a predictor to estimate rating from review length
x = [_text_length_feature(d) for d in data]  # Vector of 1 and length
y = [d['rating'] for d in data]  # Ratings
theta, residuals, rank, s = np.linalg.lstsq(x, y)  # Solve matrix equation

theta  # [3.68568136e+00   6.87371675e-05] Positive, longer reviews lead to higher ratings?


def _text_length_linear_reg(theta, review_length):
    """Represents the equation rating = θ_0 + θ_1 * length of review
    """
    return theta[0] + theta[1] * review_length


# Calculate MSE
