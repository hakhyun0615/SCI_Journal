import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import price_per_pyeong_interpolate

class RNN_Transaction_Dataset(Dataset):
    def __init__(self, data, sequence_length=5):
        data['계약년월'] = pd.to_datetime(data['계약년월'])
        interpolated_data = pd.DataFrame(data.groupby(['시군구', '단지명']).apply(price_per_pyeong_interpolate)['평단가']).reset_index().rename(columns={'level_2': '계약년월'})

        dongs_x, dongs_y = [], []
        for dong in interpolated_data['시군구'].unique():
            dong_x, dong_y = [], []
            for apartment_complex in interpolated_data[interpolated_data['시군구'] == dong]['단지명'].unique():
                filtered_interpolated_data_values = interpolated_data[interpolated_data['단지명'] == apartment_complex]['평단가'].values
                for idx in range(len(filtered_interpolated_data_values)-sequence_length):
                    apartment_complex_x = filtered_interpolated_data_values[idx:idx+sequence_length]
                    apartment_complex_y = filtered_interpolated_data_values[idx+sequence_length:idx+sequence_length+1]
                    dong_x.append(apartment_complex_x)
                    dong_y.append(apartment_complex_y)   
            dongs_x.append(dong_x)
            dongs_y.append(dong_y)

        self.dongs_x = dongs_x
        self.dongs_y = dongs_y
        self.len = len(dongs_x)

    def __getitem__(self, i):
        return torch.FloatTensor(self.dongs_x[i]), torch.FloatTensor(self.dongs_y[i])

    def __len__(self):
        return self.len