import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUAttention(nn.Module):
    def __init__(self, GRU_model, hid_dim, out_dim):
        super(GRUAttention, self).__init__()
        self.GRU_model = GRU_model
        self.fc1 = nn.Linear(hid_dim + hid_dim, hid_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hid_dim, out_dim)

    # input : hid_dim
    # hidden : 1 * num * hid_dim
    def forward(self, src, index, max_len):
        _, hidden = self.GRU_model(src)
        input = hidden[0][index]
        hidden = hidden[:,:max_len,:]
        
        query = input.unsqueeze(1)   # hid_dim, 1
        key = hidden[0]  # num * hid_dim
        value = hidden[0]  # num * hid_dim

        att_score = (key @ query)   # num, 1
        att_value = F.softmax(att_score, dim=0)  # num,1

        a = (att_value.permute(1,0) @ value).squeeze() # hid_dim
        s = torch.cat([query.squeeze(), a])  # 2hid_dim
        # s = dropout(s)
        s = self.fc1(s)  # hid_dim
        s = self.tanh(s)  # hid_dim

        y_hat = self.fc2(s) # oud_dim
        return y_hat