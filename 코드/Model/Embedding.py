from torch import nn

class Embedding(nn.Module):
    def __init__(self, encoder_dim_1, encoder_dim_2, embedding_dim, decoder_1, decoder_2):
        super(Embedding, self).__init__()

        self.encoder_1 = nn.Linear(12, encoder_dim_1)
        self.encoder_2 = nn.Linear(encoder_dim_1, encoder_dim_2)
        self.embedding = nn.Linear(encoder_dim_2, embedding_dim)
        self.decoder_1 = nn.Linear(embedding_dim, decoder_1)
        self.decoder_2 = nn.Linear(decoder_1, decoder_2)
        self.output = nn.Linear(decoder_2, 1)

    def forward(self, x):
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.embedding(x)
        x = self.decoder_1(x)
        x = self.decoder_2(x)

        return x
