import torch
import torch.nn as nn
import torch.nn.functional as F

class NLinearAttention(nn.Module):
    def __init__(self, Nlienar_model, hid_dim, out_dim, device):
        super(NLinearAttention, self).__init__()
        self.device = device

        self.Nlienar_model = Nlienar_model.to(device)
        self.fc1 = nn.Linear(hid_dim * 2, hid_dim).to(device)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hid_dim, out_dim).to(device)

    def forward(self, src, index, max_len):
        src = src.to(self.device)
        _, hidden = self.Nlienar_model(src)
        input = hidden[index].to(self.device)
        hidden = hidden[:max_len, :].to(self.device)

        query = input.unsqueeze(1)   # hid_dim, 1
        key = hidden  # num * hid_dim
        value = hidden  # num * hid_dim

        att_score = (key @ query)   # num, 1
        att_value = F.softmax(att_score, dim=0)  # num, 1

        a = (att_value.permute(1, 0) @ value).squeeze() # hid_dim
        s = torch.cat([query.squeeze(), a])  # 2hid_dim
        s = self.fc1(s)  # hid_dim
        s = self.tanh(s)  # hid_dim

        y_hat = self.fc2(s) # out_dim
        return y_hat
