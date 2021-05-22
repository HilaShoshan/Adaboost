from sklearn.model_selection import train_test_split


def get_classifiers(S_points):
    classifiers = list()  # create an empty list of classifiers


def run(points_df, labels_df, k):

    # Split the data randomly into 0.5 test (T) and 0.5 train (S)
    S_points, T_points, S_labels, T_labels = train_test_split(points_df, labels_df, test_size=0.5, random_state=1)

    H = get_classifiers(S_points)

    # for i in range(k):

