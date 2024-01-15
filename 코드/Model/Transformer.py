import math
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, emb_dim, window_size, out_dim, nhead, nlayers, dropout=0.5):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)

        self.linear =  nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(window_size, (window_size+out_dim)//2),
            nn.ReLU(),
            nn.Linear((window_size+out_dim)//2, out_dim)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    # input : num * window_size * emb_dim
    def forward(self, src, srcmask):
        src = self.pos_encoder(src)      # num * window_size * emb_dim
        hidden = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)  # num * window_size * emb_dim
        hidden = self.linear(hidden)[:,:,0]   # num * window_size
        output = self.linear2(hidden)
        return output, hidden

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask