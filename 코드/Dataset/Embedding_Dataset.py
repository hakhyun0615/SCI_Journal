import pandas as pd
import torch
from torch.utils.data import Dataset

class Embedding_Dataset(Dataset):
    def __init__(self, table_1, table_2, table_3):
        table_merge = pd.merge(table_1, table_3, how='left', on='aid')
        table_merge = pd.merge(table_merge, table_2, how='left', on='did')

        apartment = table_merge[[cols for cols in table_merge.columns if cols not in ['aid','location','name','did','year','month','call_rate','m2','price']]]
        economy = table_merge[['call_rate','m2']]
        price = table_merge[['price']]

        apartment_tensor = torch.FloatTensor(apartment.values)
        economy_tensor = torch.FloatTensor(economy.values)
        price_tensor = torch.FloatTensor(price.values)
        
        self.input_tensor = torch.cat((apartment_tensor, economy_tensor), dim=1)
        self.output_tensor = price_tensor

    def __getitem__(self, i):
        return self.input_tensor[i], self.output_tensor[i]

    def __len__(self):
        return len(self.input_tensor)