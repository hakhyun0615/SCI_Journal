import torch.nn as nn

class RNN(nn.Module):
      def __init__(self, input_size, hidden_size, output_size=1):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

      def forward(self, x):
            out, _ = self.rnn(x)
            out = self.fc(out)[0]
            return out