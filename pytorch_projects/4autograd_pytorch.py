#####################################--MANUAL CALCULATIONS--#########################################
# import numpy as np
#
# X = np.array([1, 2, 3, 4], dtype=np.float32)
# Y = np.array([2, 4, 6, 8], dtype=np.float32)
#
# w = 0.0
#
# # model prediction
# def forward(x):
#     return w*x
#
# # loss
# def loss(y, y_predicted):
#     # MSE
#     return np.mean((y_predicted-y)**2)
#
# # gradient
# #MSE = 1/n * (w*x - y)**2
# #gradient = 1/n * 2 * (w*x - y) * x
# def gradient(x, y, y_predicted):
#     return np.dot(2*x, y_predicted-y).mean()
#
# print(f"prediction before training: f(5) = {forward(5):.3f}") # 0
#
# # training
# learning_rate = 0.01
# n_iters = 12
#
# for epoch in range(n_iters):
#     # prediction
#     y_pred = forward(X)
#
#     # loss
#     l = loss(Y, y_pred)
#
#     # gradients
#     dw = gradient(X, Y, y_pred)
#
#     # update weights
#     w -= learning_rate * dw
#
#     if epoch%2 == 0:
#         print(f"epoch {epoch+1}: weight = {w:.3f}\n" +
#               f"loss = {l:.8f}")
#
# print(f"prediction after training: f(5) = {forward(5):.3f}")
#####################################--MANUAL CALCULATIONS--#########################################

##################################--AUTO GRADIENT CACLULATIONS--#####################################
# import torch
#
# # X = np.array([1, 2, 3, 4], dtype=np.float32)
# # Y = np.array([2, 4, 6, 8], dtype=np.float32)
# X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
#
# w = torch.tensor(0, dtype=torch.float32, requires_grad=True)
#
# # model prediction
# def forward(x):
#     return w*x
#
# # loss
# def loss(y, y_predicted):
#     # MSE
#     return torch.mean((y_predicted-y)**2)
#
# # gradient
# #MSE = 1/n * (w*x - y)**2
# #gradient = 1/n * 2 * (w*x - y) * x
#
#
# print(f"prediction before training: f(5) = {forward(5):.3f}") # 0
#
# # training
# learning_rate = torch.tensor(0.1)
# n_iters = 19
#
# for epoch in range(n_iters):
#     # prediction
#     y_pred = forward(X)
#
#     # loss
#     l = loss(Y, y_pred)
#
#     # backpropagation
#     l.backward() # dl/dw (calculate the derivative of loss with respects to w)
#
#     # gradients
#     dw = w.grad
#
#     # print(l, dw)
#     # update weights
#     with torch.no_grad():
#         w.sub_(learning_rate * dw)
#
#     if epoch%2 == 0:
#         print(f"epoch {epoch+1}: weight = {w:.3f}\n" +
#               f"loss = {l:.8f}")
#
#     w.grad.zero_()
#
# print(f"prediction after training: f(5) = {forward(5):.3f}")
##################################--AUTO GRADIENT CACLULATIONS--#####################################

##############################--AUTO LOSS CALC AND PERAMETER UPDATE--################################
# # general training pipeline
# # 1) Design Model (input size and output size, forward pass)
# # 2) Construct the Loss and optimizer
# # 3) Training loop
# #       - forward pass: compute prediction
# #       - backward pass: gradients
# #       - update weights
# import torch
# import torch.nn as nn
#
# # X = np.array([1, 2, 3, 4], dtype=np.float32)
# # Y = np.array([2, 4, 6, 8], dtype=np.float32)
# X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
#
# w = torch.tensor(0, dtype=torch.float32, requires_grad=True)
#
# # model prediction
# def forward(x):
#     return w*x
#
# # gradient
# #MSE = 1/n * (w*x - y)**2
# #gradient = 1/n * 2 * (w*x - y) * x
#
#
# print(f"prediction before training: f(5) = {forward(5):.3f}") # 0
#
# # training
# learning_rate = torch.tensor(0.1)
# n_iters = 19
#
# loss = nn.MSELoss() # parameters as actual y and predicted y
# optimizer = torch.optim.SGD([w], lr=learning_rate)
#
# for epoch in range(n_iters):
#     # prediction
#     y_pred = forward(X)
#
#     # loss
#     l = loss(Y, y_pred)
#
#     # backpropagation
#     l.backward() # dl/dw (calculate the derivative of loss with respects to w)
#
#     # update weights
#     optimizer.step()
#
#     if epoch%2 == 0:
#         print(f"epoch {epoch+1}: weight = {w:.3f}\n" +
#               f"loss = {l:.8f}")
#
#     optimizer.zero_grad()
#
# print(f"prediction after training: f(5) = {forward(5):.3f}")
##############################--AUTO LOSS CALC AND PERAMETER UPDATE--################################

########################################--FULL AUTOMATION--##########################################
# general training pipeline
# 1) Design Model (input size and output size, forward pass)
# 2) Construct the Loss and optimizer
# 3) Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights
import torch
from torch import nn

# X = np.array([1, 2, 3, 4], dtype=np.float32)
# Y = np.array([2, 4, 6, 8], dtype=np.float32)
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([[5], [10]], dtype=torch.float32)
n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

# model prediction - simple one layer
model = nn.Linear(input_size, output_size)

# custom model
# class LinearRegression(nn.Module):
#
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegression, self).__init__()
#         self.lin = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         return self.lin(x)
# model = LinearRegression(input_size, output_size)


# gradient
#MSE = 1/n * (w*x - y)**2
#gradient = 1/n * 2 * (w*x - y) * x


print(f"prediction before training: f(5) = {model(X_test)}") # 0

# training
learning_rate = 0.1
n_iters = 75

loss = nn.MSELoss() # loss module
                    # parameters as actual y and predicted y

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # gradient decent

for epoch in range(n_iters):
    # prediction
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # backpropagation
    l.backward() # dl/dw (calculate the derivative of loss with respects to w)

    # update weights
    optimizer.step()

    if epoch%2 == 0:
        [w, b] = model.parameters()
        print()
        print(f"epoch {epoch+1}: weight = {w[0][0].item():.3f}\n" +
              f"loss = {l:.8f}")

    optimizer.zero_grad()

print(f"prediction before training: f(5) = {model(X_test)}") # 10
########################################--FULL AUTOMATION--##########################################

