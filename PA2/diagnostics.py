import numpy as np
import ast
from collections import defaultdict
from sklearn import linear_model


def _parse_data(fname):
    """Load multiple JSON objects organized by line,
    by returning an eval each line
    """
    for l in open(fname):
        yield ast.literal_eval(l)


# Load dataset into a NumPy array
print("Reading data...")
data_list = list(_parse_data('data/beer_50000.json'))
print("Done")
data = np.array(data_list)
np.random.shuffle(data)  # Randomize order of data
training_data = data[:len(data) // 2]  # First half
testing_data = data[len(data) // 2:]  # Second half

# Intger encode categories to feature indices for dict
category_counts = defaultdict(int)
for d in data:
    category_counts[d['beer/style']] += 1
categories = [c for c in category_counts if category_counts[c] > 1000]
catID = dict(zip(list(categories), range(len(categories))))

# One hot encode
beer_oh = {}
for beer, integer in catID.items():
    ncode = [0 for _ in range(len(catID))]
    ncode[integer] = 1
    beer_oh[beer] = ncode

beer_oh
# {'American Double / Imperial IPA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 'Rauchbier': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...}


"""Begin question 1"""


def _oh_style_feature(datum):
    """Return a vector of the one hot encoding of
    a beer style. All zeros if category is not > 1000
    """
    if datum['beer/style'] in beer_oh:
        feat = beer_oh[datum['beer/style']]
    else:
        feat = [0 for _ in range(len(beer_oh))]
    return feat


def _greater_than_seven(datum):
    """Return whether a beer/ABV alcohol ABV is greater
    than 7
    """
    return 1 if datum['beer/ABV'] > 7 else 0


# Train data
x = [_oh_style_feature(d) for d in training_data]  # Our OH encodings
y = [_greater_than_seven(d) for d in training_data]  # ABV
model = linear_model.LogisticRegression(C=10, class_weight='balanced')
model.fit(x, y)

# Test data
x_testing = [_oh_style_feature(d) for d in testing_data]
y_testing = [_greater_than_seven(d) for d in testing_data]
predictions = model.predict(x_testing)


def evaluate_classifier(predictions, y):
    # Find Balanced Error Rates and Accuracy
    true_positive = 0  # Correctly guess > 7
    true_negative = 0  # Correctly guess <= 7
    false_positive = 0  # Incorrectly guess > 7
    false_negative = 0  # Incorrectly guess <= 7
    num_correct = 0
    for i in range(len(y)):
        if (predictions[i] == y[i]):
            num_correct += 1
            if (y[i] == 1):
                true_positive += 1
            else:
                true_negative += 1
        elif (y[i] == 0):
            false_positive += 1
        elif (y[i] == 1):
            false_negative += 1

    false_positive_rate = 0 if false_positive == 0 else false_positive / \
        (false_positive + true_negative)
    false_negative_rate = 0 if false_negative == 0 else false_negative / \
        (true_positive + false_negative)
    balanced_error_rate = (false_positive_rate + false_negative_rate) / 2

    accuracy = num_correct / len(y)

    return (balanced_error_rate, accuracy)


print("Question 1: (BER, accuracy) = " +
      str(evaluate_classifier(predictions, y_testing)))
# Question 1: (BER, accuracy) = (0.16164803880878556, 0.84892)


"""Begin question 2"""


reviews = [len(d['review/text']) for d in data]  # Whole dataset
max_length = np.max(reviews)  # amax


def _oh_more_feature(datum):
    """Return a vector of the one hot encoding of
    a beer style. All zeros if category is not > 1000
    Add other features like review score and length
    """
    feat = []
    if datum['beer/style'] in beer_oh:
        feat.extend(beer_oh[datum['beer/style']])
    else:
        feat.extend([0 for _ in range(len(beer_oh))])
    for feature in ['review/overall', 'review/appearance', 'review/aroma', 'review/palate', 'review/taste']:
        feat.append(datum[feature])
    feat.append(len(datum['review/text']) / max_length)
    return feat


# Train data
x = [_oh_more_feature(d) for d in training_data]  # Our OH encodings
model = linear_model.LogisticRegression(
    C=10, class_weight='balanced', max_iter=50000)
model.fit(x, y)

# Test data
x_testing = [_oh_more_feature(d) for d in testing_data]
predictions = model.predict(x_testing)

print("Question 2: (BER, accuracy) = " +
      str(evaluate_classifier(predictions, y_testing)))
# Question 2: (BER, accuracy) = (0.14524837735364052, 0.85976)


"""Begin question 3"""


testing_data = data[len(data) // 2:int(len(data) // (3/4))]  # 3rd quarter
validation_data = data[int(len(data) // (4/3)):]  # Last quarter

# Train data
x = [_oh_more_feature(d) for d in training_data]  # Our OH encodings
x_testing = [_oh_more_feature(d) for d in testing_data]
x_validation = [_oh_more_feature(d) for d in validation_data]
y_testing = [_greater_than_seven(d) for d in testing_data]
y_validation = [_greater_than_seven(d) for d in validation_data]

# Train for c in {10^−6, 10^−5, 10^−4, 10^−3} and print results for each data set
for c in [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3]:
    model = linear_model.LogisticRegression(
        C=c, class_weight='balanced', max_iter=50000)
    model.fit(x, y)

    # Test data
    predictions_training = model.predict(x)
    print("C: " + str(c))
    print("Question 3 training: (BER, accuracy) = " +
          str(evaluate_classifier(predictions, y)))
    predictions_testing = model.predict(x_testing)
    print("Question 3 testing: (BER, accuracy) = " +
          str(evaluate_classifier(predictions_testing, y_testing)))
    predictions_validation = model.predict(x_validation)
    print("Question 3 validation: (BER, accuracy) = " +
          str(evaluate_classifier(predictions_validation, y_validation)))
# C: 1e-06
# Question 3 training: (BER, accuracy) = (0.49738864932960036, 0.50888)
# Question 3 testing: (BER, accuracy) = (0.3216817274378889, 0.67348)
# Question 3 validation: (BER, accuracy) = (0.32126583725463576, 0.67368)
# C: 1e-05
# Question 3 training: (BER, accuracy) = (0.49738864932960036, 0.50888)
# Question 3 testing: (BER, accuracy) = (0.3198495204087482, 0.67548)
# Question 3 validation: (BER, accuracy) = (0.3187152077812908, 0.6764)
# C: 0.0001
# Question 3 training: (BER, accuracy) = (0.49738864932960036, 0.50888)
# Question 3 testing: (BER, accuracy) = (0.2977243047687146, 0.69824)
# Question 3 validation: (BER, accuracy) = (0.2978363107424733, 0.69792)
# C: 0.001
# Question 3 training: (BER, accuracy) = (0.49738864932960036, 0.50888)
# Question 3 testing: (BER, accuracy) = (0.19751699115182944, 0.80376)
# Question 3 validation: (BER, accuracy) = (0.19595310900242832, 0.80504)

# I would choose C = 10^-3, as it has the highest accuracy and lowest BER for testing and validation data


"""Begin question 4"""


def _oh_no_rev_length_feature(datum):
    """More features but without review length
    """
    feat = []
    if datum['beer/style'] in beer_oh:
        feat.extend(beer_oh[datum['beer/style']])
    else:
        feat.extend([0 for _ in range(len(beer_oh))])
    for feature in ['review/overall', 'review/appearance', 'review/aroma', 'review/palate', 'review/taste']:
        feat.append(datum[feature])
    return feat


def _oh_no_reviews_feature(datum):
    """More features but without reviews
    """
    feat = []
    if datum['beer/style'] in beer_oh:
        feat.extend(beer_oh[datum['beer/style']])
    else:
        feat.extend([0 for _ in range(len(beer_oh))])
    feat.append(len(datum['review/text']) / max_length)
    return feat


def _oh_no_styles_feature(datum):
    """More features but without styles
    """
    feat = []
    for feature in ['review/overall', 'review/appearance', 'review/aroma', 'review/palate', 'review/taste']:
        feat.append(datum[feature])
    feat.append(len(datum['review/text']) / max_length)
    return feat


# Train for c in {10^−6, 10^−5, 10^−4, 10^−3} and print results for each data set
print("No review length")
x = [_oh_no_rev_length_feature(d) for d in training_data]  # Our OH encodings
x_testing = [_oh_no_rev_length_feature(d) for d in testing_data]
x_validation = [_oh_no_rev_length_feature(d) for d in validation_data]
for c in [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3]:
    model = linear_model.LogisticRegression(
        C=c, class_weight='balanced', max_iter=50000)
    model.fit(x, y)

    # Test data
    predictions_training = model.predict(x)
    print("C: " + str(c))
    print("Question 4 training: (BER, accuracy) = " +
          str(evaluate_classifier(predictions, y)))
    predictions_testing = model.predict(x_testing)
    print("Question 4 testing: (BER, accuracy) = " +
          str(evaluate_classifier(predictions_testing, y_testing)))
    predictions_validation = model.predict(x_validation)
    print("Question 4 validation: (BER, accuracy) = " +
          str(evaluate_classifier(predictions_validation, y_validation)))

# Train for c in {10^−6, 10^−5, 10^−4, 10^−3} and print results for each data set
print("No reviews")
x = [_oh_no_reviews_feature(d) for d in training_data]  # Our OH encodings
x_testing = [_oh_no_reviews_feature(d) for d in testing_data]
x_validation = [_oh_no_reviews_feature(d) for d in validation_data]
for c in [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3]:
    model = linear_model.LogisticRegression(
        C=c, class_weight='balanced', max_iter=50000)
    model.fit(x, y)

    # Test data
    predictions_training = model.predict(x)
    print("C: " + str(c))
    print("Question 4 training: (BER, accuracy) = " +
          str(evaluate_classifier(predictions, y)))
    predictions_testing = model.predict(x_testing)
    print("Question 4 testing: (BER, accuracy) = " +
          str(evaluate_classifier(predictions_testing, y_testing)))
    predictions_validation = model.predict(x_validation)
    print("Question 4 validation: (BER, accuracy) = " +
          str(evaluate_classifier(predictions_validation, y_validation)))

# Train for c in {10^−6, 10^−5, 10^−4, 10^−3} and print results for each data set
print("No beer styles")
x = [_oh_no_styles_feature(d) for d in training_data]  # Our OH encodings
x_testing = [_oh_no_styles_feature(d) for d in testing_data]
x_validation = [_oh_no_styles_feature(d) for d in validation_data]
for c in [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3]:
    model = linear_model.LogisticRegression(
        C=c, class_weight='balanced', max_iter=50000)
    model.fit(x, y)

    # Test data
    predictions_training = model.predict(x)
    print("C: " + str(c))
    print("Question 4 training: (BER, accuracy) = " +
          str(evaluate_classifier(predictions, y)))
    predictions_testing = model.predict(x_testing)
    print("Question 4 testing: (BER, accuracy) = " +
          str(evaluate_classifier(predictions_testing, y_testing)))
    predictions_validation = model.predict(x_validation)
    print("Question 4 validation: (BER, accuracy) = " +
          str(evaluate_classifier(predictions_validation, y_validation)))
# No review length
# C: 1e-06
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.3165327557757349, 0.6796)
# Question 4 validation: (BER, accuracy) = (0.31927418707346444, 0.67696)
# C: 1e-05
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.3144025205852433, 0.68176)
# Question 4 validation: (BER, accuracy) = (0.31689441991168205, 0.67936)
# C: 0.0001
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.29191855043681514, 0.7048)
# Question 4 validation: (BER, accuracy) = (0.29365234845443594, 0.7032)
# C: 0.001
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.19120693136926453, 0.81004)
# Question 4 validation: (BER, accuracy) = (0.192974708952228, 0.80856)
# No reviews
# C: 1e-06
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.2690264459553762, 0.72024)
# Question 4 validation: (BER, accuracy) = (0.2706093938177439, 0.71824)
# C: 1e-05
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.16032199821131082, 0.84816)
# Question 4 validation: (BER, accuracy) = (0.16095062224006423, 0.848)
# C: 0.0001
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.16032199821131082, 0.84816)
# Question 4 validation: (BER, accuracy) = (0.16095062224006423, 0.848)
# C: 0.001
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.16032199821131082, 0.84816)
# Question 4 validation: (BER, accuracy) = (0.16095062224006423, 0.848)
# No beer styles
# C: 1e-06
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.3373876111157119, 0.6584)
# Question 4 validation: (BER, accuracy) = (0.34151103974307506, 0.6544)
# C: 1e-05
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.3370711657402383, 0.65872)
# Question 4 validation: (BER, accuracy) = (0.34128462464873544, 0.65464)
# C: 0.0001
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.33312908958259013, 0.66284)
# Question 4 validation: (BER, accuracy) = (0.3364142914492172, 0.65968)
# C: 0.001
# Question 4 training: (BER, accuracy) = (0.4988854676033005, 0.50816)
# Question 4 testing: (BER, accuracy) = (0.3180955872825419, 0.67908)
# Question 4 validation: (BER, accuracy) = (0.32050582095543956, 0.67688)
