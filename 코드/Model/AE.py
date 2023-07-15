import torch.nn as nn

class Auto(nn.Module):
      def __init__(self, input_size):
            super(Auto, self).__init__()

            # 인코더(encoder) 정의
            self.encoder = nn.Sequential(
                  nn.Linear(input_size, 32),
                  nn.ReLU(),
                  nn.Linear(32,16),
                  nn.ReLU(),
            )

            # 디코더(decoder) 정의
            self.decoder = nn.Sequential(
                  nn.Linear(16,32),
                  nn.ReLU(),
                  nn.Linear(32, input_size),
                  nn.Sigmoid()
            )

      def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded