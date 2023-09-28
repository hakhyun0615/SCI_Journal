import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import price_interpolate

class RNN_Transaction_Dataset(Dataset):
    def __init__(self, data, window_size=5):
        data['계약년월'] = pd.to_datetime(data['계약년월'].astype(str), format='%Y%m')
        interpolated_data = pd.DataFrame(data.groupby(['동', '단지']).apply(price_interpolate)['제곱미터당 거래금액(만원)']).reset_index().rename(columns={'level_2':'계약년월'})

        dongs_x, dongs_y = [], []
        for dong in interpolated_data['동'].unique():
            for apartment_complex in interpolated_data[interpolated_data['동'] == dong]['단지'].unique():
                filtered_interpolated_data_values = interpolated_data[interpolated_data['단지'] == apartment_complex]['제곱미터당 거래금액(만원)'].values
                for idx in range(len(filtered_interpolated_data_values)-window_size):
                    apartment_complex_x = filtered_interpolated_data_values[idx:idx+window_size]
                    apartment_complex_y = filtered_interpolated_data_values[idx+window_size:idx+window_size+1]
                    dongs_x.append(apartment_complex_x)
                    dongs_y.append(apartment_complex_y)     

        self.dongs_x = dongs_x
        self.dongs_y = dongs_y
        self.len = len(dongs_x)

    def __getitem__(self, i):
        return torch.FloatTensor(self.dongs_x[i]), torch.FloatTensor(self.dongs_y[i])

    def __len__(self):
        return self.len