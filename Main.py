import pandas as pd
from Adaboost import *
from statistics import mean


# create a df of points (x,y) and their labels
df = pd.read_table('rectangle.txt', delim_whitespace=True, names=('x', 'y', 'label'))
points_df = df[['x', 'y']].copy()
labels_df = df[['label']].copy()


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
    for i in range(1):  # change to 100!
        train_err_lst, test_err_lst = run(points_df, labels_df, 8)
        empirical_errors.append(train_err_lst)
        test_errors.append(test_err_lst)
    print("Empirical Errors Mean: \n", mean(empirical_errors))
    print("Test Errors Mean: \n", mean(test_errors))


if __name__ == '__main__':
    main()