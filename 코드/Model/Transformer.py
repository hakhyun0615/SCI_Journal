import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # max_len * d_model
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [0,1,2,,,,max_len] -> [[0],[1],,,,[max_len]]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))   # 크기 : 2/d
        pe[:, 0::2] = torch.sin(position * div_term)  # 각각에 0번째, 2번째, 4번째,,,는 sin 값으로
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # 1 * max_len * d_model -> max_len * 1 * d_model
        self.register_buffer('pe', pe)  # state_dict에 저장된다, 학습되지 않는다!!!!!!!!!!!!!!!!!

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)  # x와 0 을 비교해서 그 자리에 True/False 출력
    return mask
    

class TFModel(nn.Module):
    def __init__(self, iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)    # 크기는 동일, embedding 값에 크기만큼 더함

        # input : in *1 => output : in * d_model
        self.encoder = nn.Sequential(
            nn.Linear(1, d_model//2),   # weight 값을 곱해줌 => input : ir*ic, weight : ic * d/2 => output : ir*d/2
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)  # ir*d
        )

        # input : in * d => output : in * 1
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        # input : in * iw => output : in * ow
        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # torch.triu : 삼각행렬 만듬 -> transpose함으로써 역삼각행렬 생성
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  #masked_fill 함수 : 0을 -무한으로 바꿈, 1을 0으로 바꿈
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)  # iw*1 => iw*d_model
        src = self.pos_encoder(src)  # iw*d_model => iw*d_mode
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)  # iw*d_mode
        output = self.linear(output)[:,:,0]  # iw*d_mode => iw
        output = self.linear2(output)  # iw => ow
        return output