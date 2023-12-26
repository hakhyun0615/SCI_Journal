from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(12, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x