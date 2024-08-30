import torch
import torch.nn as nn

class NLinear(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.projection = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.Linear(x) 
        x = x.permute(0, 2, 1) 
        x = self.projection(x) 
        return x.squeeze(-1)