import torch
import torch.nn as nn

class GRU(nn.Module):
      def __init__(self, emb_dim, hid_dim):
            super(GRU, self).__init__()
            self.hid_dim = hid_dim
            self.gru = nn.GRU(input_size=emb_dim, hidden_size=hid_dim, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hid_dim, 1)

      # input : num * window_size * emb_dim
      def forward(self, input):
            # hidden : 1 * num * hid_dim
            hidden = torch.FloatTensor(torch.randn(1, input.shape[0], self.hid_dim)).to(input.device)
            hiddens, hidden = self.gru(input, hidden)  # hiddens : num * window_size * emb_dim
            y_hat = self.fc(hidden[0])   # num * 1  
            return y_hat, hidden
