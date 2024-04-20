import torch
import numpy as np


#####################################--TEST--#########################################

#------------------------------- CALCULATING GRADIENT ------------------------------#
# a gradient is the partial derivative of the function with respects to the values in
# the tensors it is based on. You can only print the .grad if there is only one value
# in the tensor that you used .backward() on
#
tensor1 = torch.randn(3, requires_grad=True) # if requires_grad == True, pytorch will create
#                                              # a computational graph when we do an operation
#                                              # with this tensor and any tensor that uses this
#                                              # tensor
# print(tensor1)
#
# tensor2 = tensor1 + 2
# print(tensor2) # the grad_fn is the backpropagation function
#
# # computational graph (every node, in boxes, are operations)
# # tensor1
# #        \ ___
# #         | + | ---- tensor2
# #         |___|
# #        /
# #    2
#
# tensor3 = tensor2 * tensor2 * 2
# print()
# print(tensor3)
# tensor3 = tensor3.mean()
# print(tensor3)
# print()
#
# v = torch.tensor([0.1, 1, 0.001], dtype=torch.float32)
#
# tensor3.backward() # calculates the gradient of tensor3 with respects to tensor1
# # tensor2.backward(v) # must pass the vector since there is more than one element
# #                     # in tensor2
# dx = tensor1.grad
# print(tensor1.grad)
# print()
#
# tensor = torch.tensor(3, dtype=torch.float32, requires_grad=True)
#
# tensor2 = 2*tensor + 1 # 2 * tensor^0 + 0 (partial derivative)
# tensor3 = tensor**2 - 2*tensor # 2*tensor - 2
# print(tensor2)
# print(tensor3)
# tensor2.backward()
# tensor3.backward()
# print(tensor.grad)

# # # EXTRA (how to not track gradient?)
# # x.requires_grad_(False)
# tensor1.requires_grad_(False)
# print()
# print(tensor1)
# print()

# x.detach()
ng_tensor1 = tensor1.detach()
print(ng_tensor1)
print(tensor1)
print()

# # with torch.no_grad()
# with torch.no_grad():
#     ng_tensor1 = tensor1 + 3
#     print(ng_tensor1)
#     print(tensor1)
# print(tensor1)
# # # EXTRA (how to not track gradient?)

#------------------------------- CALCULATING GRADIENT ------------------------------#

#--------------------------------- TRAINING EXAMPLES -------------------------------#
# weights = torch.ones(4, requires_grad=True)
# print(weights, "\n", sep="")

# # # TEST 1
# for epoch in range(2):
#     model_output = (weights*3).sum()
#     print(model_output)
#
#     model_output.backward() # accumulates the gradient
#
#     gradient = weights.grad
#     print(gradient)
#
#     weights.grad.zero_() # zeroing out the gradient to prevent incorrect accumulation
#     # print(gradient, "\n", sep="")
# # # TEST 1


#--------------------------------- TRAINING EXAMPLES -------------------------------#

#####################################--TEST--#########################################
