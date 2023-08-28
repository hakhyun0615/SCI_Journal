import pandas as pd
import torch
from torch.utils.data import Dataset

class NODE_Transaction_Dataset(Dataset):
    def __init__(self, data, sequence_length=5):
        data['계약년월'] = pd.to_datetime(data['계약년월']).dt.strftime('%Y%m%d')

        dongs_x, dongs_y = [], []
        for dong in data['시군구'].unique():
            for apartment_complex in data[data['시군구'] == dong]['단지명'].unique():
                filtered_data_values = data[data['단지명'] == apartment_complex]['평단가'].values
                filtered_data_times = data[data['단지명'] == apartment_complex]['계약년월'].apply(lambda x: int(x)).values 
                for idx in range(len(filtered_data_values)-sequence_length):
                    apartment_complex_and_time_x = []
                    apartment_complex_and_time_x.extend([
                        filtered_data_values[idx:idx+sequence_length],
                        filtered_data_times[idx:idx+sequence_length]
                    ])
                    apartment_complex_and_time_y = []
                    apartment_complex_and_time_y.extend([
                        filtered_data_values[idx+sequence_length:idx+sequence_length+1],
                        filtered_data_times[idx+sequence_length:idx+sequence_length+1]
                    ])
                    dongs_x.append(apartment_complex_and_time_x)
                    dongs_y.append(apartment_complex_and_time_y)

        self.dongs_x = dongs_x
        self.dongs_y = dongs_y
        self.len = len(dongs_x)

    # 부동산_x, 부동산_시점_x, 부동산_y, 부동산_시점_y 
    def __getitem__(self, i):
        return torch.FloatTensor(self.dongs_x[i][0]), torch.tensor(self.dongs_x[i][1], dtype=torch.int64), torch.FloatTensor(self.dongs_y[i][0]), torch.tensor(self.dongs_y[i][1], dtype=torch.int64)

    def __len__(self):
        return self.len