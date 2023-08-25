import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import price_per_pyeong_interpolate

class RNN_Transaction_Dataset(Dataset):
    def __init__(self, data, sequence_length=5):
        data['계약년월'] = pd.to_datetime(data['계약년월'])
        interpolated_data = pd.DataFrame(data.groupby(['시군구', '단지명']).apply(price_per_pyeong_interpolate)['평단가']).reset_index().rename(columns={'level_2': '계약년월'})
        dongs_data = []
        for dong in interpolated_data['시군구'].unique():
            dong_data = []
            for apartment_complex in interpolated_data[interpolated_data['시군구'] == dong]['단지명'].unique():
                filtered_interpolated_data_values = interpolated_data[interpolated_data['단지명'] == apartment_complex]['평단가'].values
                for idx in range(len(filtered_interpolated_data_values) - sequence_length):
                    apartment_complex_x = filtered_interpolated_data_values[idx:idx + sequence_length]
                    apartment_complex_y = filtered_interpolated_data_values[idx + sequence_length:idx + sequence_length + 1]
                    dong_data.append((apartment_complex_x, apartment_complex_y))
            dongs_data.append(dong_data)
        self.dongs_data = dongs_data
        self.len = len(dongs_data)

    # 동 하나 가져오기(따라서 batch_size도 1로 해야 함)
    def __getitem__(self, i):
        dong_data = self.dongs_data[i]
        dong_x, dong_y = [], []
        for apartment_complex_x, apartment_complex_y in dong_data:
            dong_x.append(torch.FloatTensor(apartment_complex_x))
            dong_y.append(torch.FloatTensor(apartment_complex_y))
        return dong_x, dong_y

    # 전체 동 개수
    def __len__(self):
        return self.len