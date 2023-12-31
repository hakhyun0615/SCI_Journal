import torch.nn as nn
import torch

class NLinear(nn.Module):
    def __init__(self, emb_dim, window_size):
        super(NLinear, self).__init__()
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.Linear = torch.nn.ModuleList()
        self.BatchNorm = torch.nn.ModuleList()
        for i in range(self.emb_dim):
            self.Linear.append(nn.Linear(self.window_size, 1))
            self.BatchNorm.append(nn.BatchNorm1d(1))
        self.Linear2 = nn.Linear(self.emb_dim, 1)
    
    def forward(self, x):
        seq_last = x[:,-1,:].detach()  # num * 1 * emb_dim
        for i in range(len(x)):
            x[:,i,:] = x[:,i,:] - seq_last 

        output = torch.zeros([x.size(0), self.emb_dim], dtype=x.dtype).to(x.device)
        for i in range(self.emb_dim):
            linear_output = self.Linear[i](x[:,:,i])
            bn_output = self.BatchNorm[i](linear_output)
            output[:,i] = bn_output.squeeze()

        output += seq_last.squeeze()  # num * emb_dim  
        output = self.Linear2(output)  # num * 1
        
        return output
