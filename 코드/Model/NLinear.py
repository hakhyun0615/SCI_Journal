import torch
import torch.nn as nn

class NLinear(nn.Module):
    def __init__(self, emb_dim, window_size):
        super(NLinear, self).__init__()
        self.emb_dim = emb_dim
        self.window_size = window_size

        self.Linear = nn.ModuleList([nn.Linear(window_size, 1) for _ in range(emb_dim)])
        self.BatchNorm = nn.ModuleList([nn.BatchNorm1d(1) for _ in range(emb_dim)])
        self.Linear2 = nn.Linear(emb_dim, 1)

    def forward(self, x):
        seq_last = x[:, -1, :].detach()

        for i in range(len(x)):
            x[:, i, :] -= seq_last 

        output = torch.zeros([x.size(0), self.emb_dim], dtype=x.dtype).to(x.device)

        for i in range(self.emb_dim):
            linear_out = self.Linear[i](x[:, :, i])
            batch_normalized_out = self.BatchNorm[i](linear_out)
            output[:, i] = batch_normalized_out.squeeze()

        output += seq_last.squeeze()
        output = self.Linear2(output)

        return output
