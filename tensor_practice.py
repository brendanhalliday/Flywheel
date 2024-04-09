import torch

x = torch.tensor([[4, 2, 3], [1, 5, 6]])

print(x)
g = x.max(0) # takes the maxium value from each column
            # returns the max value from each column 
            # along with it's corresponding array index

f = x.max(1) # takes the maxium value from each row
            # returns the max value from each row 
            # along with it's corresponding array index
g1 = x.max(1).indices.view()

print(type(g1))

  
# define two tensors 
A = torch.tensor(2., requires_grad=True) 
print("Tensor-A:", A) 
B = torch.tensor(5., requires_grad=False) 
print("Tensor-B:", B) 
  
# define a function using above defined 
# tensors 
x = A*B 
print("x:", x) 
  
# call the backward method 
x.backward() 
  
# print the gradients using .grad 
print("A.grad:", A.grad) 
print("B.grad:", B.grad) 
