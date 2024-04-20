import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_pred):
    accuracy_ = np.sum(y_test == y_pred) / len(y_test)
    return accuracy_

regressor = LogisticRegression(lr=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(accuracy(y_pred))

