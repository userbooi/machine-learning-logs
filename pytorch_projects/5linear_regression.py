import torch
from torch import nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 1) Design model (input size, output size, forward pass_
# 2) Construct loss and optimizer
# 3) Training loop
#       - forward pass: compute prediction and loss
#       - backpropagation: gradients
#       - update weights


# 0) data preparation
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1) # change dimensions to usable dimensions for pytorch Linear

n_samples, n_features = X.shape


# 1) model
input_size = n_features
output_size = 1 # (n_features)
model = nn.Linear(input_size, output_size)

# 2) loss function and optimizer function
learning_rate = 0.1
criterion = nn.MSELoss() # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # optimizer - gradient descent

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass - predictions, loss
    y_prediction = model(X)

    loss = criterion(y_prediction, y)

    # backpropagation
    loss.backward() # gradient

    # update weights
    optimizer.step()

    # empty gradient
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss: {loss.item():.3f}")

y_predicted = model(X).detach().numpy() # converts prediction to numpy array
                                        # in order to print

plt.scatter(X_numpy, y_numpy, color="red", s=10)
plt.plot(X_numpy, y_predicted, color="blue")
plt.title("Linear regression")
plt.show()
