import torch
import numpy as np


#####################################--TEST--#########################################

#------------------------------ SIMPLE TENSOR CREATION -----------------------------#
# od = torch.empty(3) #creates an empty 1D tensor with 3 elements
# print(od)
#
# td = torch.empty(3, 2) # creates an empty 2D tensor
# print(td)
#
# random = torch.rand(2, 2) # creates a tensor with random numbers from 0 - 1
# print(random)
#
# zero = torch.zeros(2, 3) # creates a tensor with all values as 0
# print(zero.dtype) # prints the data type of the tensor's value
#
# one = torch.ones(2, 4, dtype=torch.int) # creates a tensor with data type of int
# print(one.dtype, one)
# print("\n", one.size()) # prints the size of the tensor
#
# my_tensor = torch.tensor([2.5, 0.1]) # creates a custom tensor
# print(my_tensor)
#------------------------------ SIMPLE TENSOR CREATION -----------------------------#

#----------------------------- SIMPLE TENSOR OPERATION ----------------------------#
# TENSOR OPERATIONS WORKS LIKE MATRIX OPERATIONS

# rand_tensor1 = torch.rand(2, 2)
# rand_tensor2 = torch.rand(2, 2)
#
# print(rand_tensor1)
# print(rand_tensor2)

# # # ADDITION
# sum_tensor = torch.add(rand_tensor1, rand_tensor2) # or rand_tensor1 + rand_tensor2
# rand_tensor2.add_(rand_tensor1) # does an in-place addition (replaces rand_tensor2 with the sum)
#                                 # any pytorch function with a trailing underscore does replaces
#                                 # the targeted variable with the return value of the function
# print(sum_tensor)
# print(rand_tensor2)
# print(rand_tensor1)
# # # ADDITION

# # # SUBTRACTION
# diff_tensor = torch.sub(rand_tensor1, rand_tensor2) # or rand_tensor1 - rand_tensor2
# rand_tensor2.sub_(rand_tensor1) # does an in-place subtraction (replaces rand_tensor2 with the
#                                 # difference) any pytorch function with a trailing underscore
#                                 # does replaces the targeted variable with the return value of
#                                 # the function
# print(diff_tensor)
# print(rand_tensor2)
# print(rand_tensor1)
# # # SUBTRACTION

# # # MULTIPLICATION
# prod_tensor = torch.mul(rand_tensor1, rand_tensor2) # or rand_tensor1 * rand_tensor2
# rand_tensor2.mul_(rand_tensor1) # does an in-place multiplication (replaces rand_tensor2 with
#                                 # the product) any pytorch function with a trailing underscore
#                                 # does replaces the targeted variable with the return value of
#                                 # the function
# print(prod_tensor)
# print(rand_tensor2)
# print(rand_tensor1)
# # # MULTIPLICATION

# # # DIVISION
# quo_tensor = torch.div(rand_tensor1, rand_tensor2) # or rand_tensor1 / rand_tensor2
# rand_tensor2.div_(rand_tensor1) # does an in-place division (replaces rand_tensor2 with
#                                 # the quotient) any pytorch function with a trailing underscore
#                                 # does replaces the targeted variable with the return value of
#                                 # the function
# print(quo_tensor)
# print(rand_tensor2)
# print(rand_tensor1)
# # # DIVISION

# # # SLICING
# rand_tensor3 = torch.rand(5, 3)
# print(rand_tensor3)
# print()
# print(rand_tensor3[:, 0]) # all rows, first column
# print(rand_tensor3[1, :]) # row 2, all column
# print(rand_tensor3[1:3, 1]) # like normal slicing, row 1 to 2, column 2
# print(rand_tensor3[1:3, 1:]) # like normal slicing, row 1 to 2, column 2 to 3
# actual_non_tensor_value = rand_tensor3[4, 1].item() # the .item() method only works when
#                                                     # there is only one element in the tensor
# print(actual_non_tensor_value)
# # # SLICING

#----------------------------- SIMPLE TENSOR OPERATIONS ----------------------------#

#--------------------------------- TENSOR RESHAPING --------------------------------#
# rand_tensor4 = torch.rand(4, 4)
#
# print(rand_tensor4)
#
# reshaped1 = rand_tensor4.view(16) # creates a one dimensional tensor with 16 values
# reshaped2 = rand_tensor4.view(2, 8) # creates a two dimensional tensor with 2 rows and 8 column
# reshaped3 = rand_tensor4.view(-1, 8) # the -1 arguments tells pytorch to determine the number of
#                                      # rows if each row has 8 elements (the number of elements
#                                      # must be a divisor of the total elements)
#
# print(reshaped1)
# print(reshaped2)
# print(reshaped3)
#--------------------------------- TENSOR RESHAPING --------------------------------#

#------------------------- CONVERTING NUMPY TO TORCH TENSOR ------------------------#

# # # TENSOR TO NUMPY
# tensor1 = torch.ones(5)
# print(tensor1)
#
# np_arr = tensor1.numpy()
# print(type(np_arr))
# print(np_arr)
#
# print()
#
# tensor1.add_(torch.ones(5)) # points to the same memory location and there for change
#                             # the same memory location. Resulting in the numpy array
#                             # being changed aswell
# print(tensor1)
# print(np_arr)
# # # TENSOR TO NUMPY

# # # NUMPY TO TENSOR
np_arr = np.ones(5, dtype=np.int32)
print(np_arr)

tensor2 = torch.from_numpy(np_arr)
print(tensor2)

print()

np_arr += 1 # numpy version of adding all values by 1
print(np_arr)
print(tensor2)
# # # NUMPY TO TENSOR
#------------------------- CONVERTING NUMPY TO TORCH TENSOR ------------------------#

#####################################--TEST--#########################################