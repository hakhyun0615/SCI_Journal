import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerAttention(nn.Module):
    def __init__(self, Transformer_model, window_size, out_dim, device):
            super(TransformerAttention, self).__init__()
            self.device = device

            self.Transformer_model = Transformer_model
            self.fc1 = nn.Linear(window_size + window_size, window_size)
            self.tanh = nn.Tanh()
            self.fc2 = nn.Linear(window_size, out_dim)

    # src : num * window_size * emb_dim
    def forward(self, src, index, max_len):
            src_mask = self.Transformer_model.generate_square_subsequent_mask(src.shape[1]).to(src.device)
            output, hidden = self.Transformer_model(src, src_mask)
            input = hidden[index]  # window_size
            hidden = hidden[:max_len,:].unsqueeze(0)  # 1 * num * window_size
            
            query = input.unsqueeze(1)   # window_size, 1
            key = hidden[0]  # num * window_size
            value = hidden[0]  # num * window_size

            att_score = (key @ query)   # num, 1
            att_value = F.softmax(att_score, dim=0)  # num,1

            a = (att_value.permute(1,0) @ value).squeeze() # window_size
            s = torch.cat([query.squeeze(), a])  # window_size
            # s = dropout(s)
            s = self.fc1(s)  # window_size
            s = self.tanh(s)  # window_size

            y_hat = self.fc2(s) # oud_dim
            return y_hat