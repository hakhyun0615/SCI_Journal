import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PyraformerLayer(nn.Module):
    def __init__(self, emb_dim, nhead, dim_feedforward=2048, dropout=0.1):
        super(PyraformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(emb_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(emb_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, emb_dim)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Pyraformer(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(Pyraformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, emb_dim)
        self.layers = nn.ModuleList([PyraformerLayer(emb_dim, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.output_proj = nn.Linear(emb_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, emb_dim)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, emb_dim)

        # Pooling across the sequence dimension to reduce it
        hidden = x.mean(dim=1)  # (batch, emb_dim)

        output = self.output_proj(hidden)  # (batch, output_dim)
        return output, hidden
