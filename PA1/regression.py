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


"""Begin question 2"""


def _text_length_feature(datum):
    """Return a vector of just one feature,
    The length of the review text in characters
    """
    feat = [1, len(datum['review_text'])]
    return feat


# Train a predictor to estimate rating from review length
x = [_text_length_feature(d) for d in data]  # Vector of 1 and review length
y = [d['rating'] for d in data]  # Ratings
theta, residuals, rank, s = np.linalg.lstsq(x, y)  # Solve matrix equation

theta  # [3.68568136e+00  6.87371675e-05] Positive, longer reviews lead to higher ratings?


def _new_regression_eq(theta, review_length):
    """Represents the equation rating = θ_0 + θ_1 * (length of review)
    """
    return theta[0] + theta[1] * review_length


# Calculate MSE
sum = 0
for i in range(len(data)):
    sum += (_new_regression_eq
            (theta,
             len(data[i]['review_text'])) - y[i]) ** 2
mse = sum / len(data)  # 1.5522086622355356


"""Begin question 3"""


def _text_length_and_comments_feature(datum):
    """Return a vector of
    the length of the review text in characters
    and the number of comments
    """
    feat = [1, len(datum['review_text']), datum['n_comments']]
    return feat


# Train a predictor to estimate rating from review length
x = [_text_length_and_comments_feature(d)
     for d in data]  # Vector of 1, length, and number of comments
theta, residuals, rank, s = np.linalg.lstsq(x, y)  # Solve matrix equation


theta  # [3.68916737e+00  7.58407490e-05  -3.27928935e-02], coefficient θ_1 is different because there is a third parameter when performing matrix multiplication with its own value


def _new_regression_eq(theta, review_length, num_comments):  # Re-define
    """Represents the equation rating = θ_0 + θ_1 * (length of review) + θ_2 * (number of comments)
    """
    return theta[0] + theta[1] * review_length + theta[2] * num_comments


# Calculate MSE
sum = 0
for i in range(len(data)):
    sum += (_new_regression_eq
            (theta,
             len(data[i]['review_text']), data[i]['n_comments']) - y[i]) ** 2
mse = sum / len(data)  # 1.549835169277462, slightly better


"""Begin question 4"""

# Find maximum review length
reviews = [len(d['review_text']) for d in data]
max_length = np.max(reviews)  # amax
print(max_length)


def _text_length_poly_feature(datum, degree):
    """Return a vector of just one feature,
    The length of the review text in characters, with polynomials up to 5
    """
    length = len(datum['review_text']) / max_length
    feat = [length ** (i + 1) for i in range(degree)]
    feat.insert(0, 1)
    return feat


# Train a predictor to estimate rating from review length
x = [_text_length_poly_feature(d, 5)
     for d in data]  # Vector of 1, review length with degree 5
y = [d['rating'] for d in data]  # Ratings
theta, residuals, rank, s = np.linalg.lstsq(x, y)  # Solve matrix equation

theta
# Degree 3: [3.63659658  2.8884065  -8.48042966  6.12504475]
# 4: [3.64736873  2.20419719  -1.80763945  -11.6451833  12.21844408]
# 5: [  3.6441158  2.47396326  -5.65441081  5.55309592  -15.94637484  14.68100179]


def _new_regression_eq(theta, review_length):
    """Represents the equation rating = θ_0 + (θ_1 * (length of review)^1) + ... + (θ_n * (length of review)^n)
    """
    review_length /= max_length
    val = theta[0]
    for i in range(len(theta) - 1):
        val += theta[i + 1] * review_length ** (i + 1)
    return val


# Calculate MSE
sum = 0
for i in range(len(data)):
    sum += (_new_regression_eq
            (theta,
             len(data[i]['review_text'])) - y[i]) ** 2
mse = sum / len(data)
# Degree 3: 1.5497985323805497
# 4: 1.549629132452472
# 5: 1.5496142023298662
print(mse)
