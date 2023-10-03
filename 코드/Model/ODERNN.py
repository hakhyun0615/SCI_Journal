import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class OR_ODE_Func(nn.Module):
	
	def __init__(self, hidden_size, layer_dim):
		super(OR_ODE_Func, self).__init__()
		self.hidden_layer = nn.Linear(hidden_size, layer_dim)
		self.output_layer = nn.Linear(layer_dim, hidden_size)
		self.tanh = nn.Tanh()
		
	def forward(self, t, input):
		x = input
		x = self.hidden_layer(x)
		x = self.output_layer(x)
		x = self.tanh(x)
		return x


class ODE_RNN(nn.Module):
    def __init__(self, hidden_size, layer_dim, batch_size, atol, rtol, device):
        super(ODE_RNN, self).__init__()
        self.ode_func = OR_ODE_Func(hidden_size, layer_dim)
        self.rnn_cell = nn.GRUCell(input_size = 1, hidden_size =hidden_size)
        self.h0 = torch.zeros(batch_size, hidden_size, device = device)
        self.output_hidden = nn.Linear(hidden_size, layer_dim)
        self.output_output = nn.Linear(layer_dim, 1)
        self.output_tanh = nn.Tanh()
        self.device = device
        self.atol = atol
        self.rtol = rtol
	
    # data : batch * num   (0번째 시간, 1번째 데이터)
    # t : batch * (num+1)
    # 주의!! batch_size = 1
    def forward(self, data, t):
        output = torch.zeros(t.shape[0], t.shape[1], device = self.device)

        # h0 -> h1 구함 (h0은 0인 상태, h1은 내가 가지고 있는 값중 맨 처음 상태)
        hp = odeint(self.ode_func, self.h0, torch.tensor([0.0, t[0, 0]], device = self.device), rtol = self.rtol, atol = self.atol)[1]
        # h1 -> out1 
        out = self.output_hidden(hp)  # (batch_size, layer_dim)
        out = self.output_output(out) # (batch_size, 1)
        out = self.output_tanh(out)   # (batch_size, 1)
        output[:, 0] = out.reshape(-1)
        h = self.rnn_cell(data[0,0].reshape(-1,1), hp)
        
        for i in range(1, t.shape[1]-1):
            # hi -> hi+1
            hp = odeint(self.ode_func, h, t[0, i - 1:i + 1], rtol = self.rtol, atol = self.atol)[1]  ####### h->hp
            out = self.output_hidden(hp)
            out = self.output_output(out)
            out = self.output_tanh(out)
            output[:, i] = out.reshape(-1)
            h = self.rnn_cell(data[0,i].reshape(-1,1),hp)

        # 마지막 
        hp = odeint(self.ode_func, h, t[0, t.shape[1]-2:t.shape[1]], rtol = self.rtol, atol = self.atol)[1]  ####### h->hp
        out = self.output_hidden(hp)
        out = self.output_output(out)
        out = self.output_tanh(out)
        
        if t.shape != data.shape:
            output[:,-1] = out.reshape(-1)
        
        return output, hp