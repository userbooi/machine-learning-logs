import torch
import numpy as np


#####################################--TEST--#########################################

#---------------------------------- CALCULATING LOSS --------------------------------#
# chain rule: dz/dx = dz/dy * dy/dx
# the three steps to calculating loss:
#           1. forward pass: compute the loss
#           2. compute local gradients
#           3. backward pass (backpropagation):
#              compute dLoss / d(start_variable) using the chain rule



#---------------------------------- CALCULATING LOSS --------------------------------#

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

y_hat = w*x
loss = (y_hat-y)**2 # calculates loss with respects to w essentially
# print(loss)

loss.backward() # calculates the gradient of loss with respects to w
print(w.grad) # prints the gradient

#####################################--TEST--#########################################