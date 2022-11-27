import torch.nn as nn
import torch
import torch.nn.functional as F
batch_size = 64
vec_dim = 256
hidden_dim = 100
y = torch.rand(size=[260,64,vec_dim])
f= nn.RNN(vec_dim,hidden_dim)
out,h = f(y)
#squeeze()只会压缩维度为1时的tensor，如果这个维度上的shape不是1，那就不会压缩
lin = nn.Linear(hidden_dim,2)
o = lin(h)
o.squeeze(0)
print(y)