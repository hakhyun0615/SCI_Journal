import torch.nn as nn
import torch

# channels : 변수의 개수
class NLinear(nn.Module):
    def __init__(self, emb_dim, window_size, out_dim):
        super(NLinear, self).__init__()
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.out_dim = out_dim
        self.Linear = nn.Linear(self.window_size, self.out_dim)
        self.Linear = torch.nn.ModuleList()
        for i in range(self.emb_dim):
            self.Linear.append(torch.nn.Linear(self.window_size, self.out_dim))
        self.Linear2 = nn.Linear(self.emb_dim, 1)
    
    # input : num * window_size * emb_dim
    def forward(self, x):
        # x: [Batch, Input length]
        # detach : 역전파 하지 X, 분리시킴
        seq_last = x[:,-1,:].detach()  # num * 1 * emb_dim
        for i in range(len(x)):
            x[:,i,:] = x[:,i,:] - seq_last 
        # num * emb_dim
        output = torch.zeros([x.size(0), self.emb_dim],dtype=x.dtype).to(x.device)
        for i in range(self.emb_dim):
            output[:,i] = self.Linear[i](x[:,:,i])
        output += seq_last.squeeze()  # num * emb_dim  
        output = self.Linear2(output)  # num * 1
        
        return output