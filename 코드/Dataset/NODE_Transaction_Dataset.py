import pandas as pd
import torch
from torch.utils.data import Dataset

class NODE_Transaction_Dataset(Dataset):
    def __init__(self, data, sequence_length=5):
        dongs_x, dongs_y = [], []
        for dong in data['시군구'].unique():
            for apartment_complex in data[data['시군구'] == dong]['단지명'].unique():
                filtered_data_values = data[data['단지명'] == apartment_complex]['평단가'].values
                filtered_data_times = data[data['단지명'] == apartment_complex]['계약년월'].apply(lambda x: float((x.year-pd.Timestamp('2006-01').year)*12+(x.month-pd.Timestamp('2006-01').month))).values
                for idx in range(len(filtered_data_values)-sequence_length):
                    dongs_x.append([filtered_data_values[idx:idx+sequence_length],filtered_data_times[idx:idx+sequence_length]])
                    dongs_y.append([filtered_data_values[idx+sequence_length:idx+sequence_length+1],filtered_data_times[idx+sequence_length:idx+sequence_length+1]])

        self.dongs_x = dongs_x
        self.dongs_y = dongs_y
        self.len = len(dongs_x)

    # 부동산_x, 부동산_시점_x, 부동산_y, 부동산_시점_y 
    def __getitem__(self, i):
        return torch.FloatTensor(self.dongs_x[i][0]), torch.tensor(self.dongs_x[i][1], dtype=torch.float32), torch.FloatTensor(self.dongs_y[i][0]), torch.tensor(self.dongs_y[i][1], dtype=torch.float32)

    def __len__(self):
        return self.len