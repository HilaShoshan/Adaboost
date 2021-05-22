import pandas as pd
from Adaboost import *

df = pd.read_table('rectangle.txt', delim_whitespace=True, names=('x', 'y', 'label'))  # point (x,y) and it's label
points_df = df[['x', 'y']].copy()
labels_df = df[['label']].copy()


def main():

    print(df.head(5))
    print(df.shape)

    print("points: \n", points_df.head(5))
    print(points_df.shape)

    print("labels: \n", labels_df.head(5))
    print(labels_df.shape)

    for i in range(100):
        run(points_df, labels_df, 8)
        # print avg errors


if __name__ == '__main__':
    main()