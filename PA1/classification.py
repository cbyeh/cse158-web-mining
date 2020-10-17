import numpy as np
import ast
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
# Discard any entries that don't specify gender, 20403 left
filtered_data_list = list(d for d in data_list if 'user/gender' in d)
data = np.array(filtered_data_list)


"""Begin question 7"""


def _text_length_feature(datum):
    """Return a vector of just one feature,
    The length of the review text in characters
    """
    feat = [1, len(datum['review/text'])]
    return feat


x = [_text_length_feature(d) for d in data]  # Vector of 1 and review length
y = [d['user/gender'] for d in data]  # Genders
model = linear_model.LogisticRegression()
model.fit(x, y)
predictions = model.predict(x)


def evaluate_classifier(predictions, y):
    # Find True Positive, True Negative, False Positive, False Negative, and Balanced Error Rates
    true_positive = 0  # Correctly guessed as female
    true_negative = 0  # Correctly guessed as male
    false_positive = 0  # Incorrectly guessed as female
    false_negative = 0  # Correctly guessed as male

    for i in range(len(y)):
        if (predictions[i] == y[i] == 'Female'):
            true_positive += 1
        elif (predictions[i] == y[i] == 'Male'):
            true_negative += 1
        elif (predictions[i] != y[i] == 'Female'):
            false_positive += 1
        elif (predictions[i] != y[i] == 'Male'):
            false_negative += 1

    true_positive_rate = true_positive / len(y)
    true_negative_rate = true_negative / len(y)
    false_positive_rate = false_positive / len(y)
    false_negative_rate = false_negative / len(y)
    balanced_error_rate = (false_positive_rate + false_negative_rate) / 2

    return (true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, balanced_error_rate)


evaluate_classifier(predictions, y)
# In order of true positive rate, true negative rate, false positive rate, false negative rate, balanced error rate
# (0.0, 0.9849041807577317, 0.015095819242268294, 0.0, 0.007547909621134147)


"""Begin question 8"""

# Retrain using balanced
model = linear_model.LogisticRegression(class_weight='balanced')
model.fit(x, y)
predictions = model.predict(x)

evaluate_classifier(predictions, y)
# Note: Our classifier got worse. Our BER is higher because we are classifying a lot of men as women,
# Because most who took the survey were men
# In order of true positive rate, true negative rate, false positive rate, false negative rate, balanced error rate
# (0.00975346762730971, 0.41283144635592806, 0.0053423516149585844, 0.5720727344018036, 0.2887075430083811)


"""Begin question 9"""


def _more_features_feature(datum):
    """Return a vector of multiple features that may help
    classify a female better
    """
    feat = [1, len(datum['review/text']), float(datum['beer/beerId']),
            float(datum['beer/brewerId']), datum['review/overall'],
            datum['review/palate'], datum['review/taste'],
            datum['review/appearance'], datum['review/aroma']]
    return feat


x = [_more_features_feature(d) for d in data]
# Convert all to float
# Retrain using balanced and more features
model = linear_model.LogisticRegression(class_weight='balanced')
model.fit(x, y)
predictions = model.predict(x)

print(evaluate_classifier(predictions, y))
# Note: Performed worse for true positive
# In order of true positive rate, true negative rate, false positive rate, false negative rate, balanced error rate
# (0.0032348184090574914, 0.7855707493995981, 0.011861000833210802, 0.1993334313581336, 0.10559721609567221)
