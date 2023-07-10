import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

# ODE 함수(조정 가능)
class ODE_Func(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(ODE_Func, self).__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        out = self.elu(self.fc1(x))
        out = self.elu(self.fc2(out))
        out = self.fc3(out)
        return out

# 인코더
class NODE_Encoder(nn.Module):
      def __init__(self, input_dim, hidden_dim, latent_dim):
            super(NODE_Encoder, self).__init__()
            self.latent_dim = latent_dim

            self.rnn = nn.GRU(input_dim, hidden_dim)
            self.hid2lat = nn.Linear(hidden_dim, latent_dim)

      def forward(self, x, t):
            xt = torch.cat((x, t), dim=1)

            _, h0 = self.rnn(xt.flip((0,)))

            qz0_mean = self.hid2lat(h0[0])[:, :self.latent_dim]
            qz0_log_var = self.hid2lat(h0[0])[:, self.latent_dim:]

            return qz0_mean, qz0_log_var

# 디코더
class NODEDecoder(nn.Module):
      def __init__(self, latent_dim, hidden_dim, output_dim, device):
            super(NODEDecoder, self).__init__()
            self.relu = nn.ReLU()

            func = ODE_Func(latent_dim, hidden_dim).to(device)
            self.ode = odeint(func)
            self.l2h = nn.Linear(latent_dim, hidden_dim)
            self.h2o = nn.Linear(hidden_dim, output_dim)

      def forward(self, z0, t):
            pred_z = self.ode(z0, t, return_whole_sequence=True)

            pred_x = self.h2o(self.relu(self.l2h(pred_z)))

            return pred_x