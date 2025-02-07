import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#
# print(X)
# print(y)

# print(X_train.shape) # contains the x coordinates
# # print(type(X_train)) # numpy array
# print(X_train[0])
#
# print(y_train.shape) # contains the y coordinates
# # print(type(y_train)) # numpy array
# print(y_train[0])
#
# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

# print(y_train)
# print(X_train)

def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

mse_v = mse(y_test, predicted)
print(mse_v)

y_pred_line = regressor.predict(X)
# print(y_pred_line)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()
