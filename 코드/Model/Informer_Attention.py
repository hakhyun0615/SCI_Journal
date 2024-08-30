import torch
import torch.nn as nn
import torch.nn.functional as F

class InformerAttention(nn.Module):
    def __init__(self, Informer_model, emb_dim, out_dim, device):
        super(InformerAttention, self).__init__()
        self.device = device

        self.Informer_model = Informer_model
        self.fc1 = nn.Linear(emb_dim + emb_dim, emb_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(emb_dim, out_dim)

    def forward(self, src, index, max_len):
        output, hidden = self.Informer_model(src)

        input = hidden[index]  # emb_dim
        hidden = hidden[:max_len,:].unsqueeze(0) # 1 * num * emb_dim

        query = input.unsqueeze(1)   # emb_dim, 1
        key = hidden[0]  # num * emb_dim
        value = hidden[0]  # num * emb_dim

        att_score = (key @ query)   # num, 1
        att_value = F.softmax(att_score, dim=0)  # num,1

        a = (att_value.permute(1,0) @ value).squeeze() # emb_dim
        s = torch.cat([query.squeeze(), a])  # emb_dim
        # s = dropout(s)
        s = self.fc1(s)  # emb_dim
        s = self.tanh(s)  # emb_dim

        y_hat = self.fc2(s) # oud_dim
        return y_hat