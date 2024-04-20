import numpy as np

# logistic regression - probability
# sigmoid function is used to model logistic regression
# formula = 1/(1 + e^(-x))

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient decent
        for _ in range(self.n_iters):
            y_lin = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(y_lin)

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_lin = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(y_lin)
        y_pred_class = [1 if y > 0.5 else 0 for y in y_pred]

        return y_pred_class

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) # np.exp is the euler number (the 2.7 thing e)
