# python version: 3.7

import pandas as pd
import matplotlib.pyplot as plt
from Adaboost import *


# create a df of points (x,y) and their labels
df = pd.read_table('rectangle.txt', delim_whitespace=True, names=('x', 'y', 'label'))
points_df = df[['x', 'y']].copy()
labels_df = df[['label']].copy()


def average(lst):
    return sum(lst) / len(lst)


def plot_err(epochs, train_err, test_err):
    plt.xlabel('epochs')
    plt.ylabel('prediction error')
    plt.plot(epochs, train_err, color='blue', linewidth=3, label='train error')
    plt.plot(epochs, test_err, color='red', linewidth=3, label='test error')
    plt.legend()
    plt.show()


def main():
    """
    print(df.head(5))
    print(df.shape)

    print("points: \n", points_df.head(5))
    print(points_df.shape)

    print("labels: \n", labels_df.head(5))
    print(labels_df.shape)
    """
    empirical_errors = list()
    test_errors = list()

    # epochs errors lists for plotting
    epochs_train_errors = [0]*8  # initialize a list of zeros of size 8 (num epochs)
    epochs_test_errors = [0]*8

    iterations = 1  # change to 100!
    for i in range(iterations):
        train_err_lst, test_err_lst = run(points_df, labels_df, 8)

        print("train_err_lst: ", train_err_lst)
        print("test_err_lst: ", test_err_lst)

        empirical_errors.extend(train_err_lst)
        test_errors.extend(test_err_lst)

        epochs_train_errors = [a + b for a, b in zip(epochs_train_errors, train_err_lst)]
        epochs_test_errors = [a + b for a, b in zip(epochs_test_errors, test_err_lst)]

    print("_______________________________________________")
    print("Empirical Errors Mean: \n", round(average(empirical_errors), 2))
    print("Test Errors Mean: \n", round(average(test_errors), 2))

    # each item on the epochs errors lists is the sum of the errors in all iterations at this epoch
    # so we'll divide each one by number of iterations to get the average
    epochs_train_errors = list(map(lambda x: x / iterations, epochs_train_errors))
    epochs_test_errors = list(map(lambda x: x / iterations, epochs_test_errors))

    # Plot the change on the (average) error along the epochs to see if there is an overfitting
    plot_err([*range(8)], epochs_train_errors, epochs_test_errors)


if __name__ == '__main__':
    main()