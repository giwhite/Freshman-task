import torch.nn as nn
import torch
import torch.nn.functional as F
y = torch.rand(size=[32,2])
print(y)
y = F.softmax(y,dim = 1)
print(y)