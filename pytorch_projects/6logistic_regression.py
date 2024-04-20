import torch
from torch import nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) data preparation
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
# print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234) # automatically splitting
                                                                                                   # the data into training and
                                                                                                   # testing with the training
                                                                                                   # corresponding and testing
                                                                                                   # corresponding

# print(X_train)
# print()
# print(X_test)
# print()
# print(y_train)
# print()
# print(y_test)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1) # change dimensions to make it usable for pytorch
y_test = y_test.view(y_test.shape[0], 1) # change dimensions to make it usable for pytorch

# 1) model
#       - linear regression function (y = wx + b)
#       - slap a sigmoid function on it
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features, output_size=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, output_size)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.011
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epoch = 1000
for epoch in range(1, num_epoch+1):
    # forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    # backpropagation
    loss.backward()
    # update params
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss.item():.4f}")


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_class = y_predicted.round()
    acc = y_predicted_class.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy: {acc:.4f}")
