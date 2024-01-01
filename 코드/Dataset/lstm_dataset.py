import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class LSTM_Dataset(Dataset):
    def __init__(self, model, table_1, table_2, table_3, embedding_dim, window_size):
        model.eval()

        apartment_complexes_embedding_matrix_with_window_size = []
        apartment_complexes_price_with_window_size = []

        table_1[[cols for cols in table_1.columns if cols not in ['aid','location','name']]] = StandardScaler().fit_transform(table_1[[cols for cols in table_1.columns if cols not in ['aid','location','name']]])
        table_2[['call_rate','m2']] = StandardScaler().fit_transform(table_2[['call_rate','m2']])
        table_3[['price']] = table_3[['price']] * 0.0001

        apartment_complexes_location = table_1['location']
        apartment_complexes_name = table_1['name']
        for apartment_complex_location, apartment_complex_name in zip(apartment_complexes_location, apartment_complexes_name):
            apartment_complex_values = table_1[(table_1['name'] == apartment_complex_name) * (table_1['location'] == apartment_complex_location)][[cols for cols in table_1.columns if cols not in ['aid','location','name']]].values
            apartment_complex_tensor = torch.FloatTensor(apartment_complex_values).repeat(204, 1)
            economy_values = table_2[['call_rate','m2']].values
            economy_tensor = torch.FloatTensor(economy_values)
            encoder_input_tensor = torch.cat((apartment_complex_tensor, economy_tensor), dim=1)

            apartment_complex_embedding_matrix = np.zeros((encoder_input_tensor.shape[0], embedding_dim))
            with torch.no_grad():
                for i in range(encoder_input_tensor.shape[0]):
                    apartment_complex_embedding_vector = model.encoder(encoder_input_tensor[i].unsqueeze(0)).squeeze().numpy()
                    apartment_complex_embedding_matrix[i] = apartment_complex_embedding_vector
            apartment_complex_embedding_matrix_tensor = torch.FloatTensor(apartment_complex_embedding_matrix)

            apartment_complex_aid = table_1[(table_1['name'] == apartment_complex_name) * (table_1['location'] == apartment_complex_location)]['aid'].squeeze()
            price_values = pd.DataFrame({'did': range(0, 204)}).merge(table_3[table_3['aid'] == apartment_complex_aid][['did','price']], on='did', how='outer').fillna(0).set_index('did').values
            price_tensor = torch.FloatTensor(price_values)

            for i in range(apartment_complex_embedding_matrix_tensor.shape[0]-window_size):
                apartment_complexes_embedding_matrix_with_window_size.append(apartment_complex_embedding_matrix_tensor[i:i+window_size, :])
                apartment_complexes_price_with_window_size.append(price_tensor[i+window_size, :])

        self.apartment_complexes_embedding_matrix_with_window_size = apartment_complexes_embedding_matrix_with_window_size
        self.apartment_complexes_price_with_window_size = apartment_complexes_price_with_window_size

    def __getitem__(self, i):
        return self.apartment_complexes_embedding_matrix_with_window_size[i], self.apartment_complexes_price_with_window_size[i]
    
    def __len__(self):
        return len(self.apartment_complexes_embedding_matrix_with_window_size)