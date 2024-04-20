import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# print(X_train.shape) # training cases and coordinates
# # print(type(X_train)) # numpy array
# print(X_train[0])
#
# print(y_train.shape) # labels
# # print(type(y_train)) # numpy array
# print(y_train[0])

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor="k", s=20)
# plt.show()

predictions = KNN(k=3)
predictions.fit(X_train, y_train)
predicted_labels = predictions.predict(X_test)
accuracy = np.sum(predicted_labels == y_test) / len(y_test)
print(accuracy)
