import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(LSTM, self).__init__()   
        self.device = device
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size).to(self.device)
        self.fc = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        out, hidden = self.lstm(x)
        out = self.fc(out)
        return out, hidden