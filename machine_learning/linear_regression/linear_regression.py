import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameter
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        # print(self.weights)
        self.bias = 0

        # gradient decent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # if _ == 0:
            #     print(y_predicted)

            dw = (1/n_sample) * np.dot(X.T, (y_predicted - y))
            db = (1/n_sample) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        # print(np.dot(X, self.weights))

        return y_predicted
