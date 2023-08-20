import torch.nn as nn
import torch

# channels : 변수의 개수
class NLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(NLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Linear = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        # x: [Batch, Input length]
        # detach : 역전파 하지 X, 분리시킴
        seq_last = x[:,-1].detach()
        for i in range(len(x)):
            x[i] = x[i] - seq_last[i]
        output = self.Linear(x)
        x = output
        for i in range(len(x)):
            x[i] = x[i] + seq_last[i]
        return x 