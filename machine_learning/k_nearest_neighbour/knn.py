import numpy as np
from collections import Counter

# k nearest neighbour (KNN) is a classification algorithm used to predict the label
# of a test case according to the distance between the test case and the other data
# samples. It uses (x1 - x2)**2 + (y1 - y2)**2 to calculate the distance of the k
# nearest values around the test case and finds the mode of the labels in the k
# nearest values. the mode label is used as a prediction for the label of the test
# case

# predicting discrete values


def euclid_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [euclid_distance(x, x_train) for x_train in self.X_train]

        # print(distances)
        k_indices = np.argsort(distances)[:self.k]
        # print(k_indices)
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1) # returns a tuple containing the common label and the
                                                               # amount of times it appeared
        return most_common[0][0]
