import numpy as np
import ast
import matplotlib.pyplot as plt


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


"""Begin question 1"""

# Plot star ratings and relationship with length
x = [len(d['review_text']) for d in data]
y = [d['rating'] for d in data]  # Ratings
plt.scatter(x, y)
# plt.show()

# Number of 0, 1, ..., 5 star ratings
zero_star_ratings = len([rating for rating in y if rating == 0])  # 326
one_star_ratings = len([rating for rating in y if rating == 1])  # 286
two_star_ratings = len([rating for rating in y if rating == 2])  # 778
three_star_ratings = len([rating for rating in y if rating == 3])  # 2113
four_star_ratings = len([rating for rating in y if rating == 4])  # 3265
five_star_ratings = len([rating for rating in y if rating == 5])  # 3232


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


"""Begin question 5"""

# Split the data
np.random.shuffle(data)  # Randomize order of data
training_data = data[:len(data) // 2]  # First half
testing_data = data[len(data) // 2:]  # Second half

# Train a predictor to estimate rating from review length
x = [_text_length_poly_feature(d, 5)
     for d in training_data]  # Vector of 1, review length with degree 5
y = [d['rating'] for d in training_data]  # Ratings
theta, residuals, rank, s = np.linalg.lstsq(x, y)  # Solve matrix equation

# Degree 3: [3.65398022  2.85740482  -8.6349972   7.03106877]
# 4: [3.65538388  2.70313039  -4.77004286  -8.0607224  11.03884403]
# 5: [3.66726222  2.44067015  -12.01072447  41.08045065  -67.57729633  35.98020285]

# Calculate MSE
sum_training = 0
sum_testing = 0
y_testing = [d['rating'] for d in testing_data]
for i in range(len(training_data)):
    sum_training += (_new_regression_eq
                     (theta,
                      len(training_data[i]['review_text'])) - y[i]) ** 2
    sum_testing += (_new_regression_eq
                    (theta,
                     len(testing_data[i]['review_text'])) - y[i]) ** 2
mse_training = sum_training / len(training_data)
mse_testing = sum_testing / len(testing_data)
# Training and testing error (varies due to random datasets):
# Note that for these results, the dataset was re-randomized
# Degree 3: 1.5516115542413098 (training), 1.5659240213323633 (testing)
# 4: 1.5604723081811585 (training), 1.5792144817011526 (testing)
# 5: 1.5200201662189214 (training), 1.5308161745889985 (testing)
