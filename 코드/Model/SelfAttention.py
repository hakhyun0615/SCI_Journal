import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# embedding을 어떻게 더 할 지 고민!!!!!!!!!!!!!!
# Q,K,V 모두 batch * seq_len * data_size
# max_size = 최대 단지 개수
# pred_len = 결과의 크기(만약 가격 1개만 알고싶으면 =1)
# input : batch_size*max_size*hidden_state
class Self_attention(nn.Module):
    def __init__(self, max_size, data_size, pred_len):
        super(Self_attention, self).__init__()
        self.max_size = max_size
        self.data_size = data_size
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.data_size, self.pred_len)
            
    def calculate_attention(self, query, key, value, max_size):
        mask = torch.zeros(key.shape)
        mask[-2] = max_size - key[-2]
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)
    
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, seq_len, seq_len)
        out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k)
        out = torch.cat((out,mask), dim=1)
        return out
    
    def forward(self, data):
        # attention 계산
        attention = self.calculate_attention(data,data,data,self.max_size)
        output = torch.zeros([data.size(0),self.max_size,self.pred_len],dtype=data.dtype).to(data.device)
        # 
        for i in range(data.shape[-2]):
            output[:,i,:] = self.Linear(attention[:,i,:])
        return output
    
    
# input : batch_size, max_len, window_size
class Attention_LSTM(nn.Module):
    # data_size : 원래 데이터 사이즈
    # hidden_size : lstm 통해 나온 사이즈
    # pred_len : 1
    def __init__(self, LSTM, max_size, pred_len, hidden_size=256):
        super(Attention_LSTM, self).__init__()
        self.LSTM = LSTM
        self.Attention = Self_attention(max_size, hidden_size, pred_len)
        
        self.max_size = max_size
        self.hidden_size = hidden_size
        
    def forward(self, data):
        return_data = torch.zeros((data.shape[0],self.max_size,self.hidden_size))
        
        # return_data에 embedding vector 넣기
        # 만약 거래가 안된 적이 존재한다면 그 단지는 무시
        for i in range(self.max_size):
            if 0 in data[:,i,:]:
                continue
            return_data[:,i,:] = self.LSTM(data[:,i,:])
            
        outs = self.Attention(return_data)
        return outs
        