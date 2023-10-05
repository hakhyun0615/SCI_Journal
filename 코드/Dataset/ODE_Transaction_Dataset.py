import pandas as pd
import torch
from torch.utils.data import Dataset

class ODE_Transaction_Dataset(Dataset):
    def __init__(self, data, window_size=5):
        data['계약년월'] = pd.to_datetime(data['계약년월'].astype(str), format='%Y%m')
        
        dongs_x, dongs_y = [], []
        for dong in data['동'].unique():
            for apartment_complex in data[data['동'] == dong]['단지'].unique():
                    filtered_data = data[(data['동'] == dong)*(data['단지'] == apartment_complex)]
                    filtered_data_values = filtered_data['제곱미터당 거래금액(만원)'].values
                    filtered_data_times = filtered_data['계약년월'].apply(lambda x: float((x.year-pd.Timestamp('2006-01').year)*12+(x.month-pd.Timestamp('2006-01').month)+1)).values
                    for idx in range(len(filtered_data_values)-window_size):
                        dongs_x.append([filtered_data_values[idx:idx+window_size],filtered_data_times[idx:idx+window_size]])
                        dongs_y.append([filtered_data_values[idx+window_size:idx+window_size+1],filtered_data_times[idx+window_size:idx+window_size+1]])
        self.dongs_x = dongs_x
        self.dongs_y = dongs_y
        self.len = len(dongs_x)

    # 부동산_x, 부동산_시점_x, 부동산_y, 부동산_시점_y 
    def __getitem__(self, i):
        return torch.FloatTensor(self.dongs_x[i][0]), torch.FloatTensor(self.dongs_x[i][1]), torch.FloatTensor(self.dongs_y[i][0]), torch.FloatTensor(self.dongs_y[i][1])

    def __len__(self):
        return self.len