import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.random

x = torch.randn(10, 3)
y = torch.randn(10, 2)  


linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)


criterion = nn.MSELoss()
optimizer = optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)

loss = criterion(pred, y)
print('loss before step: ', loss.item())
loss.backward()

optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print('loss after step: ', loss.item())
