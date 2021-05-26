from sklearn.model_selection import train_test_split
from LineClassifier import *

import math


def get_classifiers(S_points):

    classifiers = list()  # create an empty list of line classifiers
    num_lines = S_points.shape[0]

    # add to the list all the possible lines from the points in the train data
    for i in range(num_lines-1):
        for j in range(i+1, num_lines):
            x_i = S_points.iloc[i].loc['x']
            y_i = S_points.iloc[i].loc['y']
            x_j = S_points.iloc[j].loc['x']
            y_j = S_points.iloc[j].loc['y']
            classifiers.append(LineClassifier(x_i, y_i, x_j, y_j))
            classifiers.append(LineClassifier(x_i, y_i, x_j, y_j, False))

    return classifiers


def compute_error(t, best_lines, lines_weights, points, labels):
    n = points.shape[0]
    sum_err = 0
    for i in range(n):
        indicator = 0
        x_i = points.iloc[i].loc['x']
        y_i = points.iloc[i].loc['y']
        label_i = labels.iloc[i].loc['label']
        if label_i != compute_Ht(t, best_lines, lines_weights, x_i, y_i):  # an error in classification
            indicator = 1
        sum_err += indicator
    return sum_err/n


def compute_Ht(t, best_lines, lines_weights, x, y):
    sigma = 0
    for i in range(t+1):
        result = best_lines[i].get_classification(x, y)
        sigma += result*lines_weights[i]
    if sigma < 0:
        return -1
    elif sigma > 0:
        return 1
    else:
        return 0


def run(points_df, labels_df, k):

    # Split the data randomly into 0.5 test (T) and 0.5 train (S)
    S_points, T_points, S_labels, T_labels = train_test_split(points_df, labels_df, test_size=0.5, shuffle=True)

    H = get_classifiers(S_points)

    best_lines = list()
    lines_weights = list()
    n = S_points.shape[0]  # number of points on the train set
    weights = [1/n] * n  # Initialize point weights to 1/n

    train_err_lst = list()
    test_err_lst = list()

    # identify the k most important lines h_i and their respective weights
    for t in range(k):

        min_weighted_error = float('inf')
        for h in H:
            weighted_error = 0
            for i in range(n):
                indicator = 0  # indicator for the event ℎ(point_𝑖) ≠ label_𝑖
                x_i = S_points.iloc[i].loc['x']
                y_i = S_points.iloc[i].loc['y']
                label_i = S_labels.iloc[i].loc['label']
                if h.get_classification(x_i, y_i) != label_i:
                    indicator = 1
                weighted_error += weights[i]*indicator
            if weighted_error < min_weighted_error:
                min_weighted_error = weighted_error
                h_t = h  # the classifier with min weighted error

        best_lines.append(h_t)

        # find classifier weight based on its error
        alpha_t = (1/2) * math.log((1-min_weighted_error)/min_weighted_error)
        lines_weights.append(alpha_t)

        # The empirical error of Ht on the training set
        train_err_lst.append(compute_error(t, best_lines, lines_weights, S_points, S_labels))
        # The true error of Ht on the test set
        test_err_lst.append(compute_error(t, best_lines, lines_weights, T_points, T_labels))

        # Update point weights
        z_t = 0  # normalization factor
        for i in range(n):
            x_i = S_points.iloc[i].loc['x']
            y_i = S_points.iloc[i].loc['y']
            label_i = S_labels.iloc[i].loc['label']
            weights[i] = weights[i] * math.exp(-alpha_t*h_t.get_classification(x_i, y_i)*label_i)
            z_t += weights[i]  # summing the weights
        weights = list(map(lambda x: x/z_t, weights))  # normalization

    return train_err_lst, test_err_lst
