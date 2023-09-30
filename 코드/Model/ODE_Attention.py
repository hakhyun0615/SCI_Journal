import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from Model.ODERNN import *


class ODE_Attention(nn.Module):
    def __init__(self, max_size, hidden_size, pred_len,  latent, device, one_size=200):
        super(ODE_Attention, self).__init__()
        self.max_size = max_size
        self.pred_len = pred_len
        self.Linear = nn.Linear(hidden_size, self.pred_len)
        self.Q_W = ODE_RNN(hidden_size,latent,1,one_size,one_size,device)
        self.K_W = ODE_RNN(hidden_size,latent,1,one_size,one_size,device)
        self.V_W = ODE_RNN(hidden_size,latent,1,one_size,one_size,device)
            
    def calculate_attention(self, query, key, value):
        d = query.shape[-1]
        attention_score = torch.matmul(query, key.permute(0,2,1)) # Q x K^T, (n_batch, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d)
    
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, seq_len, seq_len)
        out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k)
        return out


    # tran_data : batch * 단지 개수 * 데이터 크기
    # tran_data_label : 실제 정답이 있는 데이터(실제 거래 된 데이터)
    # tran_data_label 크기 : batch * 단지 개수
    def forward(self, eco_data, tran_data, tran_data_label, time_x, time_y, alpha, eco_model):
        # 경제데이터
        eco_vector = eco_model(eco_data)
        
        # 시간 데이터 합성
        time_f = torch.cat([time_x,time_y],axis=1)
        
        # 부동산 가격 데이터 : 실제 가격이 있는 곳만 답 있음, 없는 곳은 0
        output = torch.zeros(tran_data_label.shape)
        
        # Q,K,V 생성
        Q = torch.zeros(tran_data.shape)
        K = torch.zeros(tran_data.shape)
        V = torch.zeros(tran_data.shape)
        
        for i in range(len(tran_data.shape[1])):
            Q[:,i,:] = self.Q_W(tran_data[:,i,:],time_f[:,i,:]) + eco_vector
            K[:,i,:] = self.K_W(tran_data[:,i,:],time_f[:,i,:]) + eco_vector
            V[:,i,:] = self.V_W(tran_data[:,i,:],time_f[:,i,:]) + eco_vector
        
        # mask 씌우기
        mask = tran_data[:,:,-1]
        mask = torch.where(mask == 0, mask, torch.ones_like(mask))
        
        Q = Q * mask.repeat(2,Q.shape[-1])
        K = K * mask.repeat(2,Q.shape[-1])
        V = V * mask.repeat(2,Q.shape[-1])
        
        # attention 계산
        attention = self.calculate_attention(Q,K,V)
        
        # 정답 구하기
        for i in range(len(tran_data_label)):  # mask값이 1이 있는데까지만 계산
            if tran_data_label[0,i,0] == 0:
                break
            if tran_data_label == 0:
                continue
            output[i] = self.Linear(attention[:,i,:])
        
        return output