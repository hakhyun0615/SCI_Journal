import pandas as pd
import torch
from torch.utils.data import Dataset

class ODE_Transaction_Dataset(Dataset):
    def __init__(self, data, sequence_length=5):
        data['계약년월'] = pd.to_datetime(data['계약년월']).dt.strftime('%Y%m%d') ###

        dongs_x, dongs_x_time, dongs_y, dongs_y_time = [], [], [], []
        for dong in data['시군구'].unique():
            dong_x, dong_x_time, dong_y, dong_y_time = [], [], [], []
            for apartment_complex in data[data['시군구'] == dong]['단지명'].unique():
                filtered_data_values = data[data['단지명'] == apartment_complex]['평단가'].values
                filtered_data_times = data[data['단지명'] == apartment_complex]['계약년월'].apply(lambda x: int(x)).values 
                for idx in range(len(filtered_data_values)-sequence_length):
                    apartment_complex_x = filtered_data_values[idx:idx+sequence_length]
                    apartment_complex_x_time = filtered_data_times[idx:idx+sequence_length]
                    apartment_complex_y = filtered_data_values[idx+sequence_length:idx+sequence_length+1]
                    apartment_complex_y_time = filtered_data_times[idx+sequence_length:idx+sequence_length+1]
                    dong_x.append(apartment_complex_x)
                    dong_x_time.append(apartment_complex_x_time)
                    dong_y.append(apartment_complex_y)
                    dong_y_time.append(apartment_complex_y_time)
            dongs_x.append(dong_x)
            dongs_x_time.append(dong_x_time)
            dongs_y.append(dong_y)
            dongs_y_time.append(dong_y_time)

        self.dongs_x = dongs_x
        self.dongs_x_time = dongs_x_time
        self.dongs_y = dongs_y
        self.dongs_y_time = dongs_y_time
        self.len = len(dongs_x)

    def __getitem__(self, i):
        return torch.FloatTensor(self.dongs_x[i]), torch.tensor(self.dongs_x_time[i], dtype=torch.int64), torch.FloatTensor(self.dongs_y[i]), torch.tensor(self.dongs_y_time[i], dtype=torch.int64)

    def __len__(self):
        return self.len