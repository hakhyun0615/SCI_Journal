import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class Embedding_Dataset(Dataset):
    def __init__(self, table_1, table_2, table_3):
        table_merge = pd.merge(table_1, table_3, how='left', on='aid')
        table_merge = pd.merge(table_merge, table_2, how='left', on='did')

        apartment = table_merge[[cols for cols in table_merge.columns if cols not in ['aid','location','name','did','year','month','call_rate','m2','price']]]
        economy = table_merge[['call_rate','m2']]
        price = table_merge[['price']]

        apartment_values = apartment.values
        economy_values = economy.values
        price_values = price.values
        
        input_values = np.concatenate((apartment_values, economy_values), axis=1)
        input_values = StandardScaler().fit_transform(input_values)
        output_values = price_values

        self.input_tensor = torch.FloatTensor(input_values)
        self.output_tensor = torch.FloatTensor(output_values)

    def __getitem__(self, i):
        return self.input_tensor[i], self.output_tensor[i]

    def __len__(self):
        return len(self.input_tensor)
