import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Attention_Dataset(Dataset):
    def __init__(self, table_1, table_2, table_3, apartment_complex_name, embedding_dim, window_size):
        apartment_complex = table_1[table_1['name'] == apartment_complex_name][[cols for cols in table_1.columns if cols not in ['aid','location','name']]]
        apartment_complex_tensor = torch.FloatTensor(apartment_complex.values).repeat(204, 1)
        economy = table_2[['call_rate','m2']]
        economy_tensor = torch.FloatTensor(economy.values)
        encoder_input_tensor = torch.cat((apartment_complex_tensor, economy_tensor), dim=1)

        embedding_vectors = np.zeros((encoder_input_tensor.size(0), embedding_dim))
        with torch.no_grad():
            for i in range(encoder_input_tensor.size(0)):
                embedding_vector = model.encoder(encoder_input_tensor[i].unsqueeze(0)).squeeze().numpy()
                embedding_vectors[i] = embedding_vector
        self.embedding_vectors_tensor = torch.FloatTensor(embedding_vectors)

        apartment_complex_aid = table_1[table_1['name'] == apartment_complex_name]['aid'].squeeze()
        price = pd.DataFrame({'did': range(0, 204)}).merge(table_3[table_3['aid'] == apartment_complex_aid][['did','price']], on='did', how='outer').fillna(0).set_index('did')
        self.price_tensor = torch.FloatTensor(price.values)

        self.window_size = window_size

    def __getitem__(self, i):
        return self.embedding_vectors_tensor[i:i+self.window_size, :], self.price_tensor[i+self.window_size, :]
    
    def __len__(self):
        return len(self.embedding_vectors_tensor) - self.window_size + 1