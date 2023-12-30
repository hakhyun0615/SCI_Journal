from torch import nn

class Embedding(nn.Module):
    def __init__(self, encoder_dim_1, encoder_dim_2, encoder_dim_3, embedding_dim, decoder_1, decoder_2, decoder_3):
        super(Embedding, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(12, encoder_dim_1),
            nn.ReLU(True),
            nn.Linear(encoder_dim_1, encoder_dim_2),
            nn.ReLU(True),
            nn.Linear(encoder_dim_2, encoder_dim_3),
            nn.ReLU(True),
            nn.Linear(encoder_dim_3, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, decoder_1),
            nn.ReLU(True),
            nn.Linear(decoder_1, decoder_2),
            nn.ReLU(True),
            nn.Linear(decoder_2, decoder_3),
            nn.ReLU(True),
            nn.Linear(decoder_3, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
